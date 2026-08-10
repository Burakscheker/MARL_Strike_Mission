"""n-adim getiri biriktirici — Strike_Mission.md §11.14.

NEDEN: olculdu (2026-08-07), uzun egitim politikayi BOZUYOR:
    VDN  surv_ratio  ep1000 %4.23  ->  ep3000 %2.80
    QMIX             ep1000 %0.00  ->  ep3000 %0.00
ve egitim boyunca olum 1.09 -> 1.91, adim 2269 -> 923. Kritik ayrinti:
EPSILON DUSTUKCE olum ARTIYOR, yani bozulma kesif gurultusunden degil
OGRENILEN politikadan geliyor. Intihar kapisi kapali (olculdu: olmek -88 <
oyalanmak -78), yani ayri bir mekanizma.

En olasi sebep 1-ADIM TD BOOTSTRAP'i: gamma=0.9998 ve 2800 adimlik
episode'larda etkin ufuk 5000 adim. Tek adimlik hedef
    y = r + gamma * max_a Q(s', a)
odul sinyalini episode boyunca ancak adim adim geri tasir ve her adimda
Q'nun kendi hatasini yeniden girdi olarak alir; uzun ufukta bu hata birikir.

n-adim getiri bootstrap payini AZALTIR:
    y = sum_{k=0}^{n-1} gamma^k r_{t+k}  +  gamma^n * max_a Q(s_{t+n}, a)
n buyudukce hedef Monte Carlo'ya yaklasir (n -> episode sonu = tam MC,
bootstrap YOK). Tolga'nin PPO'su GAE(lambda~0.95) kullaniyor ve o da
MC'ye bizim 1-adim TD'mizden cok daha yakin — onun egrisi yukselirken
bizimkinin bozulmasi tam bu farktan olabilir. Bu dosya o hipotezi test
edilebilir kiliyor.

DURUSTLUK NOTU: bu DUZELTILMEMIS (uncorrected) n-adim getiri. Ara odullerin
davranis politikasindan toplanmasi, hedef politika greedy oldugu icin
teorik bir yanlilik yaratir. Rainbow dahil pratikteki standart budur ve
genelde ise yarar, ama "teorik olarak dogru" degil — importance sampling
ya da Retrace bunu duzeltir, biz eklemedik.

KULLANIM: her algoritmanin push() imzasi farkli oldugu icin hangi alanin
odul, hangisinin done, hangilerinin SON gecisten alinacagi ACIKCA
belirtilir (asagidaki SPEC'ler). Sessiz indeks hatasi olmasin diye
kopyalanmis kod yerine tek jenerik sinif kullaniliyor.
"""
from collections import deque

# (reward_idx, done_idx, from_last)
#   from_last = n-adim penceresinin SON gecisinden alinacak alanlar
#               (next_obs, done, next_mask, next_state); geri kalani ILK
#               gecisten gelir (s_t, a_t ve QMIX'te state_t).
#
# DQN/IQL : (obs, action, reward, next_obs, done, next_mask)
SPEC_DQN = (2, 4, (3, 4, 5))
# VDN     : (obs1, a1, obs2, a2, r, next1, next2, done, nm1, nm2)
SPEC_VDN = (4, 7, (5, 6, 7, 8, 9))
# QMIX    : VDN + (state, next_state) -> state ILK'ten, next_state SON'dan
SPEC_QMIX = (4, 7, (5, 6, 7, 8, 9, 11))


class NStepAccumulator:
    """Gecisleri biriktirip n-adim getirili tek gecis uretir.

    push() bir LISTE dondurur: [(gecis, gamma_n), ...]. Cogu adimda 0 ya da
    1 eleman; episode sonunda kalan kisa pencereler bosaltildigi icin birden
    fazla olabilir.

    gamma_n = gamma^k (k = pencerenin GERCEK uzunlugu). Episode sonundaki
    kisa pencerelerde k < n olur ve hedefte gamma^n kullanmak YANLIS olurdu;
    bu yuzden ufuk gecisle BIRLIKTE tasinir (bkz. buffer'lardaki gamma_n
    sutunu). gamma=0.9998'de fark kucuk ama sessizce yanlis olmasindansa
    dogru olsun.
    """

    def __init__(self, n: int, gamma: float, spec):
        self.n = max(1, int(n))
        self.gamma = gamma
        self.reward_idx, self.done_idx, self.from_last = spec
        self.buf: deque = deque()

    def reset(self):
        """Episode sinirinda cagrilir. Normalde done=True zaten bosaltir, ama
        IQL'de bir ajanin episode'u KESILME (truncation) ile bitebiliyor ve o
        durumda done=False geliyor — bosaltilmazsa pencere BIR SONRAKI
        episode'a sizar ve iki episode'un odulleri toplanir."""
        self.buf.clear()

    def push(self, transition):
        self.buf.append(transition)
        out = []
        if len(self.buf) >= self.n:
            out.append(self._emit())
            self.buf.popleft()
        if bool(transition[self.done_idx]):
            while self.buf:
                out.append(self._emit())
                self.buf.popleft()
        return out

    def flush(self):
        """Kalan pencereleri (kisa ufuklu) bosalt ve dondur."""
        out = []
        while self.buf:
            out.append(self._emit())
            self.buf.popleft()
        return out

    def _emit(self):
        first, last = self.buf[0], self.buf[-1]
        acc, g = 0.0, 1.0
        for tr in self.buf:
            acc += g * float(tr[self.reward_idx])
            g *= self.gamma                  # dongu sonunda g = gamma^k
        out = list(first)
        out[self.reward_idx] = acc
        for i in self.from_last:
            out[i] = last[i]
        return tuple(out), g
