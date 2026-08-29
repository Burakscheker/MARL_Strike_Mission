"""MARL-Pathfinding'in EGITILMIS agirliklarini bu projeye tasima.

NEDEN CALISIYOR (sekil uyumu): CNNQNet'in parametre sayisi AdaptiveAvgPool2d
sayesinde PATCH_SIZE'dan bagimsiz; sekle bagli tek iki sey OBS_CHANNELS ve
N_SCALARS. Ikisi de config.py'de MARL-Pathfinding'inkiyle AYNI tutuldu (2 ve
16), dolayisiyla state_dict BIREBIR yukleniyor.

NEDEN ANLAMLI (semantik uyum): gozlem SLOT SLOT hizalandi —

    kanal 0   : orada "yasak bolge/duvar"     -> burada "tehlike haritasi"
                (ikisi de: buyuk deger = buradan kacin)
    kanal 1   : orada "kendi izi"             -> burada "kendi izi" (ayni)
    skalar 5-7: hedefe dx/dy/uzaklik          -> ayni
    skalar 8-10: diger ajana dx/dy/uzaklik    -> ayni
    skalar 11 : BFS mesafesi (kendi hucre)    -> risk-farkinda mesafe
    skalar12-15: BFS komsu FARKI (+1 yakin)   -> risk-mesafe komsu farki (+1 yakin)

Yani onceden egitilmis ag "hedefe dx/dy yonunde git, kanal 0'daki bolgelerden
kac, komsu-farki pozitif olan yone yonel" davranisini ZATEN biliyor. Burada
ogrenmesi gereken YENI sey, plandaki ifadeyle, "olme kalma muhabbeti":
tehlikenin ikili degil OLASILIKSAL olmasi ve olumun kalici bir terminal olmasi.

DURUST UYARI (olcume tabi, varsayim degil): kaynak model gamma=0.99 ve ~100
adimlik episode'larla egitildi; burada gamma=0.9998 ve ~2000 adim. Q
degerlerinin OLCEGI ~1/(1-gamma) ile buyudugu icin son katmanin ciktilari
bastan yanlis olcekte olacak. Ozellik cikarici (conv + skalar kodlayici)
transfer olur, deger olcegi olmaz. Bu yuzden --resume-head-reset secenegi var:
govdeyi yukler, son katmani sifirdan baslatir. Hangisinin daha iyi oldugu
OLCULECEK, tahmin edilmeyecek.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from config import PATHFINDING_CKPT_DIR

# Hangi algoritma icin hangi kaynak dosya(lar). MARL-Pathfinding'in
# runs/ckpt/ klasorundeki isimlendirme.
# SIRA ONEMLI — en uyumludan en uyumsuza.
# OLCULDU: `vdn_final.pt` MARL-Pathfinding'in 11-SKALARLI eski doneminden
# kalma (scalar_enc.0.weight = (128,11)), bizim 16 skalarimizla uyusmuyor ve
# ag govdesinin en kritik iki katmani (scalar_enc + head girisi) transfer
# olmuyordu. `*_bfs_*` ve `*_obs_hard` etiketleri 16-skalarli (BFS mesafe
# skalarlari eklendikten sonraki) donemden — once onlar denenmeli.
DEFAULT_SOURCE = {
    "vdn": ["vdn_bfs_medium.pt", "vdn_obs_hard.pt", "vdn_bfs_hard.pt",
            "vdn_final.pt"],
    "qmix": ["qmix_bfs_medium.pt", "qmix_obs_hard.pt", "qmix_bfs_hard.pt",
             "qmix_final.pt"],
}

# Kaynak checkpoint'in bizim gozlem boyutumuzla uyumlu olup olmadigini
# ANLAYAN tensor. Skalar sayisi degisirse bu katmanin sekli degisir.
_PROBE = "scalar_enc.0.weight"


def _scalar_dim(sd: dict) -> int | None:
    w = sd.get(_PROBE)
    return None if w is None else int(w.shape[1])


def resolve_source(algo: str, ckpt_dir: str = PATHFINDING_CKPT_DIR) -> str | None:
    """Algoritmaya uygun, GOZLEM BOYUTU UYAN ilk checkpoint'i bul.

    Sadece "dosya var mi"ya bakmak yetmiyor: ayni klasorde farkli gozlem
    surumlerinden kalma checkpoint'ler bir arada duruyor ve uyumsuz olani
    secmek transferin en degerli kismini (skalar kodlayici + head girisi)
    sessizce dusuruyor. Once TAM UYAN aranir, yoksa ilk var olana dusulur.
    """
    from config import N_SCALARS
    candidates = []
    for name in DEFAULT_SOURCE.get(algo, []):
        p = os.path.join(ckpt_dir, name)
        if not os.path.exists(p):
            continue
        candidates.append(p)
        try:
            ckpt = torch.load(p, map_location="cpu")
        except Exception:
            continue
        sd = ckpt.get("online", ckpt.get("online1"))
        if sd is not None and _scalar_dim(sd) == N_SCALARS:
            return p
    if candidates:
        print(f"UYARI: {algo} icin {N_SCALARS} skalarli checkpoint yok; "
              f"kismi transferle {os.path.basename(candidates[0])} kullanilacak")
        return candidates[0]
    return None


def load_matching(module: nn.Module, src: dict, label: str,
                  head_reset: bool = False) -> dict:
    """src state_dict'inden SEKLI UYAN tensorleri module'e yukle.

    Uymayanlar sessizce atlanmaz — sayilari ve isimleri raporlanir. Kismi
    transferin ne kadar kismi oldugunu bilmeden "resume ettim" demek, bu
    projedeki en kolay kendini kandirma yolu olurdu.
    """
    tgt = module.state_dict()
    loaded, skipped_shape, skipped_missing, reset = [], [], [], []
    for k, v in tgt.items():
        if k not in src:
            skipped_missing.append(k)
            continue
        if src[k].shape != v.shape:
            skipped_shape.append(f"{k} {tuple(src[k].shape)}->{tuple(v.shape)}")
            continue
        # head'in SON Linear'i (Q ciktisi) — deger olcegi transfer olmuyor
        if head_reset and k.startswith("head.4"):
            reset.append(k)
            continue
        tgt[k] = src[k]
        loaded.append(k)
    module.load_state_dict(tgt)
    report = {"label": label, "loaded": len(loaded), "total": len(tgt),
              "shape_mismatch": skipped_shape, "missing": skipped_missing,
              "reset": reset}
    return report


def print_report(reports: list[dict], source_path: str):
    print(f"\n--- transfer: {source_path}")
    for r in reports:
        pct = 100.0 * r["loaded"] / max(1, r["total"])
        print(f"  {r['label']:<10} {r['loaded']}/{r['total']} tensor yuklendi "
              f"(%{pct:.0f})")
        if r["reset"]:
            print(f"    sifirlanan (head): {', '.join(r['reset'])}")
        for k in r["shape_mismatch"]:
            print(f"    SEKIL UYUSMADI  : {k}")
        for k in r["missing"]:
            print(f"    KAYNAKTA YOK    : {k}")
    print("---\n")


def resume_vdn(agent, path: str, head_reset: bool = False):
    from config import AGENT_1, AGENT_2
    ckpt = torch.load(path, map_location="cpu")
    reports = []
    for a, key, lab in ((AGENT_1, "online1", "A1"), (AGENT_2, "online2", "A2")):
        src = ckpt.get(key)
        if src is None:
            print(f"  UYARI: {path} icinde '{key}' yok, {lab} sifirdan basliyor")
            continue
        reports.append(load_matching(agent.online[a], src, lab, head_reset))
        agent.target[a].load_state_dict(agent.online[a].state_dict())
    print_report(reports, path)


def resume_qmix(agent, path: str, head_reset: bool = False,
                load_mixer: bool = True):
    from config import AGENT_1, AGENT_2
    ckpt = torch.load(path, map_location="cpu")
    reports = []
    for a, key, lab in ((AGENT_1, "online1", "A1"), (AGENT_2, "online2", "A2")):
        src = ckpt.get(key)
        if src is None:
            continue
        reports.append(load_matching(agent.online[a], src, lab, head_reset))
        agent.target[a].load_state_dict(agent.online[a].state_dict())
    if load_mixer and "mixer" in ckpt:
        # Mixer'i yuklemek TARTISMALI: agirliklari GLOBAL STATE'ten uretiyor ve
        # state'in anlami degisti (orada yasak-bolge, burada tehlike). Yine de
        # sekil uyuyor ve monotonluk yapisi ayni; --no-mixer-transfer ile
        # kapatilip etkisi olculebilir.
        reports.append(load_matching(agent.mixer, ckpt["mixer"], "mixer", False))
        agent.mixer_target.load_state_dict(agent.mixer.state_dict())
    print_report(reports, path)


def resume(algo: str, agent_or_agents, path: str, head_reset: bool = False,
           load_mixer: bool = True):
    if algo == "vdn":
        resume_vdn(agent_or_agents, path, head_reset)
    elif algo == "qmix":
        resume_qmix(agent_or_agents, path, head_reset, load_mixer)
    else:
        raise ValueError(algo)
