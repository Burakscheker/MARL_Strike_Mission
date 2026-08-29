"""Ag mimarileri.

MLP: 5x5'in orijinal mimarisi (~35k parametre). Kucuk N'de duz vektoru
katmanlar arasinda dolastirmak yeterliydi.

CNNQNet: 100x100 gibi buyuk gridler icin. Duz-MLP'de ilk katman GRID_N^2 ile
BUYUR (5*100*100=50.000 girdi -> 128 gizli = 6.4M parametre SADECE ilk
katmanda, 4+ agli VDN/QMIX'te bu katlanir). CNN + kademeli stride'li
downsample + AdaptiveAvgPool2d ile parametre sayisi GRID_N'den BAGIMSIZ kalir
ve (PLAN §Asama 9 stretch hedefi) farkli grid boyutlarina da tasinabilir.
"""
import torch
import torch.nn as nn

from config import (CNN_CHANNELS, CNN_POOL_SIZE, HIDDEN, N_ACTIONS, N_SCALARS,
                    OBS_CHANNELS, PATCH_SIZE, SCALAR_EMBED)


class DeterministicAdaptiveAvgPool2d(nn.Module):
    """nn.AdaptiveAvgPool2d yerine — 2026-08-27, dis inceleme onerisi.

    BULGU: torch.use_deterministic_algorithms(True) acikken bile
    adaptive_avg_pool2d_backward_cuda'nin DETERMINISTIK bir CUDA karsiligi
    YOK (PyTorch sessizce/uyararak non-deterministik kernele duser). Bu
    oturumda AYNI --seed ile GPU'da iki kez tekrarlanan bir egitim
    TAMAMEN FARKLI sonuc uretmisti (ep200 takim %50 vs %22) — supheli tek
    nokta tam olarak buydu.

    COZUM: PyTorch'un adaptive-pooling bolme formulunu (istart=floor(i*in/
    out), iend=ceil((i+1)*in/out)) MANUEL dilimleme+ortalama ile yeniden
    uygular — hepsi deterministik ops. Satir-sonra-sutun ortalamasi
    MATEMATIKSEL OLARAK 2D dikdortgen ortalamaya ESIT (ortalama ayrisir),
    yani SAYISAL DEGER (ULP duzeyinde toplama sirasi disinda) degismez —
    ESKI checkpoint'lerin agirliklari GECERLI kalir, sadece hesaplama YOLU
    degisiyor.
    """

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size
        self._bins_cache: dict[tuple[int, int], tuple[list[int], list[int]]] = {}

    def _bins(self, in_size: int, out_size: int):
        key = (in_size, out_size)
        b = self._bins_cache.get(key)
        if b is None:
            starts = [(i * in_size) // out_size for i in range(out_size)]
            ends = [-(-((i + 1) * in_size) // out_size) for i in range(out_size)]  # ceil
            b = (starts, ends)
            self._bins_cache[key] = b
        return b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        o = self.output_size
        rs, re = self._bins(h, o)
        cs, ce = self._bins(w, o)
        rows = torch.cat([x[:, :, rs[i]:re[i], :].mean(dim=2, keepdim=True)
                          for i in range(o)], dim=2)
        return torch.cat([rows[:, :, :, cs[j]:ce[j]].mean(dim=3, keepdim=True)
                          for j in range(o)], dim=3)


class MLP(nn.Module):
    """Q agi: obs -> her aksiyon icin Q degeri.

    5x5 gridde ~35k parametre; CPU'da rahat egitilir.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNQNet(nn.Module):
    """Buyuk gridler icin Q agi.

    Girdi FORMATI degismiyor: env.observe()'un urettigi AYNI duz vektor
    [spatial.ravel(), scalars] — sadece ag ICINDE yeniden sekillendiriliyor.
    Boylece ReplayBuffer/JointReplayBuffer/MixerReplayBuffer hic degismeden
    kullanilabiliyor, degisen sadece forward()'un ic islemesi.

    stride-2 conv katmanlariyla spatial boyut kademeli kuculur (PATCH_SIZE=21
    icin 21->11->6), sonra AdaptiveAvgPool2d(pool_size) ile SABIT boyuta
    indirgenir — bu, parametre sayisini PATCH_SIZE'dan bagimsizlastirir VE
    kaba konumsal bilgiyi (global average pooling'in aksine) korur.

    SKALAR DALI (bkz. config.py SCALAR_EMBED notu): skalarlar ham haliyle
    CNN ciktisina eklenirse 523 girdinin 11'i olup kayboluyordu. Artik once
    kucuk bir MLP ile genisletiliyor, HAM hali de skip olarak korunuyor.
    """

    def __init__(self, channels: int, grid_n: int, n_scalars: int, n_actions: int,
                 conv_channels: tuple[int, ...] = (16, 32, 32),
                 pool_size: int = 6, hidden: int = HIDDEN,
                 scalar_embed: int = SCALAR_EMBED, dueling: bool = False):
        super().__init__()
        self.channels = channels
        self.grid_n = grid_n
        self.n_scalars = n_scalars
        self.spatial_size = channels * grid_n * grid_n
        self.dueling = dueling

        layers = []
        in_c = channels
        for out_c in conv_channels:
            layers += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                      nn.ReLU()]
            in_c = out_c
        self.conv = nn.Sequential(*layers)
        self.pool = DeterministicAdaptiveAvgPool2d(pool_size)

        self.scalar_enc = nn.Sequential(
            nn.Linear(n_scalars, scalar_embed), nn.ReLU(),
            nn.Linear(scalar_embed, scalar_embed), nn.ReLU(),
        )

        flat_dim = in_c * pool_size * pool_size
        in_head = flat_dim + scalar_embed + n_scalars
        if not dueling:
            self.head = nn.Sequential(
                # + n_scalars: ham skalar skip baglantisi (dx/dy -> Q dogrudan yol)
                nn.Linear(in_head, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )
        else:
            # DUELING (Wang ve ark. 2016, 2026-08-26 dis inceleme onerisi):
            # BC->RL fine-tune denemelerinde COKEN sey hep AYNIYDI — Q'nun
            # MUTLAK OLCEGI (V) egitim ilerledikce buyurken (bkz. config.py
            # §11.14, q_mean 17->37) aksiyonlar ARASI SIRALAMA (hangisi daha
            # iyi, action-gap 0.05-0.1) siliniyordu. Tek gövdeli baslikta
            # (yukaridaki 'if not dueling' dali) bu iki bilgi AYNI agirliklarda
            # karisik kodlanir. Dueling govde-sonrasi paylasilan TEK gizli
            # katmandan sonra V(s) (skaler) ve A(s,a) (n_actions) icin AYRI
            # dallara boler; Q(s,a)=V(s)+(A(s,a)-mean_a A(s,a)) — olcek (V)
            # ve siralama (A) MIMARI OLARAK ayristigi icin TD guncellemesi
            # V'yi degistirirken A'nin OGRENDIGI SIRALAMAYI ezmesi zorlasir.
            self.trunk = nn.Sequential(nn.Linear(in_head, hidden), nn.ReLU())
            self.value_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.adv_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x[:, :self.spatial_size].view(-1, self.channels, self.grid_n, self.grid_n)
        scalars = x[:, self.spatial_size:]
        h = self.conv(spatial)
        h = self.pool(h).flatten(1)
        h = torch.cat([h, self.scalar_enc(scalars), scalars], dim=1)
        if not self.dueling:
            return self.head(h)
        h = self.trunk(h)
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + (a - a.mean(dim=1, keepdim=True))


def build_qnet(n_actions: int = N_ACTIONS, dueling: bool = False) -> nn.Module:
    """Butun ajanlarin (DQN/IQL/VDN/QMIX) kullandigi TEK fabrika.

    grid_n=PATCH_SIZE (GRID_N DEGIL): env.observe() artik tam gridi degil
    ajanin YEREL penceresini donduruyor (bkz. config.py OBS_CHANNELS notu),
    ag da o pencere boyutuna gore sekilleniyor.

    dueling=False (varsayilan): ESKI mimariyle BIREBIR ayni (parametre
    isimleri dahil) — mevcut checkpoint'ler etkilenmez.
    """
    return CNNQNet(OBS_CHANNELS, PATCH_SIZE, N_SCALARS, n_actions,
                   conv_channels=CNN_CHANNELS, pool_size=CNN_POOL_SIZE,
                   dueling=dueling)


NEG_INF = -1e9      # -inf yerine: maskeli softmax/max'ta NaN uretmez


def masked_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gecersiz aksiyonlari cok buyuk negatif degere it.

    Gercek -inf kullanmiyoruz: terminal gecislerde tum aksiyonlar maskeliyse
    -inf * 0 = NaN cikar ve gradyan sessizce bozulur.
    """
    return q.masked_fill(mask <= 0, NEG_INF)
