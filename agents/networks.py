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
                 scalar_embed: int = SCALAR_EMBED):
        super().__init__()
        self.channels = channels
        self.grid_n = grid_n
        self.n_scalars = n_scalars
        self.spatial_size = channels * grid_n * grid_n

        layers = []
        in_c = channels
        for out_c in conv_channels:
            layers += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                      nn.ReLU()]
            in_c = out_c
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

        self.scalar_enc = nn.Sequential(
            nn.Linear(n_scalars, scalar_embed), nn.ReLU(),
            nn.Linear(scalar_embed, scalar_embed), nn.ReLU(),
        )

        flat_dim = in_c * pool_size * pool_size
        self.head = nn.Sequential(
            # + n_scalars: ham skalar skip baglantisi (dx/dy -> Q dogrudan yol)
            nn.Linear(flat_dim + scalar_embed + n_scalars, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x[:, :self.spatial_size].view(-1, self.channels, self.grid_n, self.grid_n)
        scalars = x[:, self.spatial_size:]
        h = self.conv(spatial)
        h = self.pool(h).flatten(1)
        h = torch.cat([h, self.scalar_enc(scalars), scalars], dim=1)
        return self.head(h)


def build_qnet(n_actions: int = N_ACTIONS) -> nn.Module:
    """Butun ajanlarin (DQN/IQL/VDN/QMIX) kullandigi TEK fabrika.

    grid_n=PATCH_SIZE (GRID_N DEGIL): env.observe() artik tam gridi degil
    ajanin YEREL penceresini donduruyor (bkz. config.py OBS_CHANNELS notu),
    ag da o pencere boyutuna gore sekilleniyor.
    """
    return CNNQNet(OBS_CHANNELS, PATCH_SIZE, N_SCALARS, n_actions,
                   conv_channels=CNN_CHANNELS, pool_size=CNN_POOL_SIZE)


NEG_INF = -1e9      # -inf yerine: maskeli softmax/max'ta NaN uretmez


def masked_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gecersiz aksiyonlari cok buyuk negatif degere it.

    Gercek -inf kullanmiyoruz: terminal gecislerde tum aksiyonlar maskeliyse
    -inf * 0 = NaN cikar ve gradyan sessizce bozulur.
    """
    return q.masked_fill(mask <= 0, NEG_INF)
