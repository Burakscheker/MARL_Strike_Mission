"""n-adim getiri biriktiricisinin dogrulugu — agents/nstep.py.

Kritik iki sart:
  1. N_STEP=1 ESKI DAVRANISLA BIREBIR AYNI olmali (yoksa tum onceki
     sonuclarimiz kiyaslanamaz hale gelir).
  2. n>1'de getiri sum gamma^k r_k, ufuk gamma^k ve "kuyruk" alanlari SON
     gecisten gelmeli; episode sonundaki KISA pencereler de dogru ufukla
     cikmali (gamma^n degil gamma^k).
"""
import numpy as np

from agents.nstep import SPEC_DQN, SPEC_VDN, NStepAccumulator

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    print(f"  [{'GECTI' if cond else 'KALDI'}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        FAIL += 1


# DQN gecisi: (obs, action, reward, next_obs, done, next_mask)
def tr(i, r, done=False):
    return (f"o{i}", i, r, f"o{i+1}", done, f"m{i+1}")


def test_n1_identical():
    """n=1: her gecis ANINDA ve degistirilmeden cikmali, gamma_n = gamma."""
    g = 0.9
    acc = NStepAccumulator(1, g, SPEC_DQN)
    outs = []
    for i, r in enumerate([1.0, 2.0, 3.0]):
        outs += acc.push(tr(i, r, done=(i == 2)))
    check("n=1 gecis sayisi degismedi", len(outs) == 3, f"{len(outs)}")
    ok = all(o[0] == tr(i, [1.0, 2.0, 3.0][i], done=(i == 2)) for i, o in enumerate(outs))
    check("n=1 gecisler AYNEN korunuyor", ok)
    check("n=1 ufuk = gamma", all(abs(o[1] - g) < 1e-12 for o in outs),
          f"{[round(o[1],4) for o in outs]}")


def test_n3_returns():
    g = 0.5
    acc = NStepAccumulator(3, g, SPEC_DQN)
    rs = [1.0, 2.0, 4.0, 8.0, 16.0]
    outs = []
    for i, r in enumerate(rs):
        outs += acc.push(tr(i, r, done=(i == len(rs) - 1)))

    # ilk cikan: t=0'dan 3 adim -> r0 + g r1 + g^2 r2 = 1 + 1 + 1 = 3
    first = outs[0]
    check("n=3 getiri dogru", abs(first[0][2] - 3.0) < 1e-9, f"{first[0][2]}")
    check("n=3 ufuk gamma^3", abs(first[1] - g ** 3) < 1e-12, f"{first[1]}")
    check("bas alanlar ILK gecisten", first[0][0] == "o0" and first[0][1] == 0)
    check("kuyruk alanlari SON gecisten",
          first[0][3] == "o3" and first[0][5] == "m3", f"{first[0][3]},{first[0][5]}")

    # toplam cikan gecis sayisi = adim sayisi (her t icin bir hedef)
    check("her adim icin bir hedef uretildi", len(outs) == len(rs), f"{len(outs)}")

    # SON gecis: sadece kendisi -> getiri r4, ufuk gamma^1, done=True
    last = outs[-1]
    check("episode sonu kisa pencere: getiri", abs(last[0][2] - 16.0) < 1e-9, f"{last[0][2]}")
    check("episode sonu kisa pencere: ufuk gamma^1 (gamma^n DEGIL)",
          abs(last[1] - g) < 1e-12, f"{last[1]}")
    check("episode sonu done=True tasindi", last[0][4] is True)


def test_no_leak_between_episodes():
    """done=False ile biten (kesilme) bir episode'dan sonra flush() sizintiyi
    onlemeli — yoksa iki episode'un odulleri ayni getiride toplanir."""
    g = 0.9
    acc = NStepAccumulator(5, g, SPEC_DQN)
    for i, r in enumerate([1.0, 1.0, 1.0]):      # 3 adim, done YOK
        acc.push(tr(i, r))
    flushed = acc.flush()
    check("flush kalan pencereleri bosaltti", len(flushed) == 3, f"{len(flushed)}")
    check("flush sonrasi tampon bos", len(acc.buf) == 0)
    outs = acc.push(tr(99, 5.0, done=True))
    check("yeni episode onceki odulle KARISMIYOR",
          abs(outs[0][0][2] - 5.0) < 1e-9, f"{outs[0][0][2]}")


def test_vdn_spec():
    """VDN imzasinda odul/done/kuyruk indeksleri dogru mu."""
    g = 0.5
    acc = NStepAccumulator(2, g, SPEC_VDN)
    # (obs1, a1, obs2, a2, r, next1, next2, done, nm1, nm2)
    t0 = ("A0", 1, "B0", 2, 10.0, "A1", "B1", False, "n1", "n2")
    t1 = ("A1", 3, "B1", 4, 20.0, "A2", "B2", True, "N1", "N2")
    outs = acc.push(t0) + acc.push(t1)
    o = outs[0][0]
    check("VDN getiri = r0 + g*r1", abs(o[4] - (10.0 + g * 20.0)) < 1e-9, f"{o[4]}")
    check("VDN bas = ILK gecis", o[0] == "A0" and o[1] == 1)
    check("VDN kuyruk = SON gecis", o[5] == "A2" and o[7] is True and o[8] == "N1")


if __name__ == "__main__":
    print("=== n-adim getiri ===")
    test_n1_identical()
    test_n3_returns()
    test_no_leak_between_episodes()
    test_vdn_spec()
    print(f"\n{'TUM TESTLER GECTI' if FAIL == 0 else str(FAIL) + ' TEST KALDI'}")
    raise SystemExit(1 if FAIL else 0)
