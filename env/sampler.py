"""Rastgele radar yerlesimi — Strike_Mission.md §11.

Burak'in tarifi (2026-08-06): "40 tane olucak random yerlerde olusucaklar
her episode farkli bi haritada olucak trainde de testde de, radarlarin
merkezleri farkli olucak ama cakisabilir, mesela merkezleri arasinda 5 fark
olabilir ustuste biner alanlari".

Yani KISIT YOK: merkezler uniform, minimum mesafe sarti aranmiyor, ortusme
serbest. Bu bilincli — ortusen radarlar seviye-tabanli risk kuralinda
(§11.3) tek bir buyuk birlesik bolgeye donusuyor ve harita "40 ayri engel"
yerine "birkac buyuk tehlike kitlesi" haline geliyor.

EGITIM/TEST TOHUM AYRIMI (asiri ogrenmeye karsi asil savunma):
Egitim haritalari `0 .. TRAIN_SEED_MAX` araligindaki tohumlardan, test
haritalari `EVAL_SEED_BASE ..` araligindan uretilir. Araliklar KESISMEZ,
yani ajanin egitimde gordugu bir haritanin testte tekrar cikma olasiligi
sifir. Harita ezberlemek imkansiz oldugu icin "test basarisi" gercekten
genellemeyi olcer.
"""
from __future__ import annotations

import numpy as np

from config import (EVAL_N_MAPS, EVAL_SEED_BASE, GRID_N, N_RADAR,
                    TRAIN_SEED_MAX)

Cell = tuple[int, int]


def sample_radars(n_radar: int = N_RADAR, rng: np.random.Generator | None = None,
                  n: int = GRID_N) -> tuple[Cell, ...]:
    """n_radar adet uniform radar merkezi (row, col). Cakisma serbest."""
    rng = np.random.default_rng() if rng is None else rng
    rows = rng.integers(0, n, size=n_radar)
    cols = rng.integers(0, n, size=n_radar)
    return tuple((int(r), int(c)) for r, c in zip(rows, cols))


def train_map_seed(rng: np.random.Generator) -> int:
    """Egitim icin harita tohumu — test araligiyla KESISMEZ."""
    return int(rng.integers(0, TRAIN_SEED_MAX))


def eval_map_seeds(n_maps: int = EVAL_N_MAPS) -> list[int]:
    """Degerlendirmenin ORTAK harita seti.

    Her algoritma (IQL/VDN/QMIX) ve her baseline AYNI haritalarda olculmeli;
    yoksa aradaki fark algoritmadan mi harita sansindan mi geldigi
    ayirt edilemez — rastgele haritada bu fark 10 kata kadar cikiyor
    (oracle tavani ortalama %32, medyan %7.2).
    """
    return [EVAL_SEED_BASE + i for i in range(n_maps)]


def curriculum_n_radar(episode: int, total_episodes: int) -> int:
    """Egitim ilerledikce radar sayisini rampala (bkz. config.CURRICULUM_*)."""
    from config import (CURRICULUM_FRAC, CURRICULUM_RADAR_END,
                        CURRICULUM_RADAR_START)
    if total_episodes <= 0:
        return CURRICULUM_RADAR_END
    frac = min(1.0, episode / max(1.0, CURRICULUM_FRAC * total_episodes))
    span = CURRICULUM_RADAR_END - CURRICULUM_RADAR_START
    return int(round(CURRICULUM_RADAR_START + frac * span))
