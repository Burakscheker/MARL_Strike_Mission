"""Episode calistiricilari — IQL / VDN / QMIX icin ortak dongu.

Egitim (train=True) ve degerlendirme (train=False, eps=0) AYNI fonksiyonu
kullanir; iki kod yolunun birbirinden sapmasi MARL-Pathfinding'de bir hata
kaynagiydi, burada bastan onleniyor.

ES ZAMANLI ORTAMIN SADELESTIRDIGI SEY: MARL-Pathfinding'de her adimda SADECE
bir ajan oynuyordu ve "golge NOOP" ile digerinin Q'su toplama sokuluyordu;
faz sinirlarini takip eden (own_done / phase_timeout) ekstra bir mantik
gerekiyordu. Burada iki ucak da HER adimda gercek aksiyon uretir, terminal
olanlar NOOP'a duser — VDN/QMIX'in joint transition'i dogal olarak olusur.
"""
from typing import Optional

from config import AGENT_1, AGENT_2, NOOP
from env.strike_env import StrikeMissionEnv


def _act_all(agents_or_agent, obs, env, train: bool, shared: bool):
    """Iki ucak icin aksiyon uret. shared=True ise tek ajan nesnesi (VDN/QMIX,
    act() agent_id ister), False ise ajan sozlugu (IQL)."""
    eps = None if train else 0.0
    acts = {}
    for a in (AGENT_1, AGENT_2):
        mask = env.action_mask(a)
        if env.terminal(a):
            acts[a] = NOOP                     # maske zaten sadece NOOP acik
        elif shared:
            acts[a] = agents_or_agent.act(a, obs[a], mask, eps=eps)
        else:
            acts[a] = agents_or_agent[a].act(obs[a], mask, eps=eps)
    return acts


def play_episode(env: StrikeMissionEnv, agents: dict, train: bool,
                 config=None, reset_kwargs: Optional[dict] = None) -> tuple[dict, dict]:
    """IQL: iki BAGIMSIZ DQN, her biri SADECE info["r_ind"][agent]'i gorur.

    Takim odulu (r_team) hic kullanilmaz — IQL'in tanimi bu. Bir ucak terminal
    olduktan sonra ARTIK push etmez (kendi episode'u bitti); digeri devam eder.
    """
    # reset_kwargs: rastgele haritada map_seed / n_radar (curriculum) buradan
    # gecer. Egitim ve degerlendirme AYNI fonksiyonu kullandigi icin harita
    # secimi de tek yerden yonetilmis olur.
    obs = env.reset(config=config, **(reset_kwargs or {}))
    losses = {AGENT_1: [], AGENT_2: []}
    info: dict = {}
    done = False

    while not done:
        was_terminal = {a: env.terminal(a) for a in (AGENT_1, AGENT_2)}
        acts = _act_all(agents, obs, env, train, shared=False)
        next_obs, _r_team, done, info = env.step(acts)

        if train:
            for a in (AGENT_1, AGENT_2):
                if was_terminal[a]:
                    continue                   # bu ucagin episode'u zaten bitti
                own_done = env.terminal(a)
                # Timeout GERCEK terminal degil (time-limit bootstrapping):
                # deger 0 degildir, bootstrap devam etmeli.
                is_trunc = done and info.get("timeout", False) and not own_done
                push_done = own_done and not is_trunc
                next_mask = env.physical_mask(a)
                agents[a].push(obs[a], acts[a], info["r_ind"][a],
                               next_obs[a], push_done, next_mask)
                loss = agents[a].learn()
                if loss is not None:
                    losses[a].append(loss)

        obs = next_obs

    # n-adim penceresini episode sinirinda bosalt (bkz. agents/nstep.py).
    # Olmazsa pencere bir sonraki episode'a sizar ve iki ayri episode'un
    # odulleri ayni getiride toplanir.
    if train:
        for a in (AGENT_1, AGENT_2):
            agents[a].end_episode()

    return info, losses


def play_episode_vdn(env: StrikeMissionEnv, agent, train: bool,
                     config=None, reset_kwargs: Optional[dict] = None,
                     health_probe: Optional[list] = None) -> tuple[dict, list]:
    """VDN: Q_tot = Q_1 + Q_2, TEK TD hatasi ikisine BIRDEN geri yayilir.

    play_episode ile TEK mimari fark: iki ucagin (obs, aksiyon) cifti AYNI
    joint transition'a yazilir — terminal olan ucagin NOOP'u DAHIL. Kredi
    kanali tam burada: terminal A1'in Q(obs,NOOP)'u, A2 ucmaya devam ederken
    degisir ve gradyan A1'in agina akar.

    health_probe (bos liste) verilirse, bir ucak terminal olduktan SONRA onun
    Q(obs, NOOP) degerlerini biriktirir. Bu degerin SABIT KALMASI, terminal
    ucagin gozleminin guncellenmedigini gosterir — MARL-Pathfinding'de "en
    sinsi bug" buydu, burada da tek saglik kontrolu bu.
    """
    # reset_kwargs: rastgele haritada map_seed / n_radar (curriculum) buradan
    # gecer. Egitim ve degerlendirme AYNI fonksiyonu kullandigi icin harita
    # secimi de tek yerden yonetilmis olur.
    obs = env.reset(config=config, **(reset_kwargs or {}))
    losses: list = []
    info: dict = {}
    done = False

    while not done:
        acts = _act_all(agent, obs, env, train, shared=True)

        if health_probe is not None:
            for a in (AGENT_1, AGENT_2):
                if env.terminal(a) and not env.done:
                    health_probe.append(agent.q_value(a, obs[a], NOOP))
                    break

        next_obs, r_team, done, info = env.step(acts)

        if train:
            is_trunc = done and info.get("timeout", False)
            push_done = done and not is_trunc
            nm1 = env.physical_mask(AGENT_1)
            nm2 = env.physical_mask(AGENT_2)
            agent.push(obs[AGENT_1], acts[AGENT_1], obs[AGENT_2], acts[AGENT_2],
                       r_team, next_obs[AGENT_1], next_obs[AGENT_2],
                       push_done, nm1, nm2)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        obs = next_obs

    if train:
        agent.end_episode()          # n-adim penceresini bosalt (nstep.py)

    return info, losses


def play_episode_qmix(env: StrikeMissionEnv, agent, train: bool,
                      config=None, reset_kwargs: Optional[dict] = None,
                      health_probe: Optional[list] = None) -> tuple[dict, list]:
    """QMIX: play_episode_vdn ile TEK fark — her joint transition'a GLOBAL
    STATE (t ve t+1) eklenir; mixer'in hypernetwork'u agirliklarini ondan uretir."""
    # reset_kwargs: rastgele haritada map_seed / n_radar (curriculum) buradan
    # gecer. Egitim ve degerlendirme AYNI fonksiyonu kullandigi icin harita
    # secimi de tek yerden yonetilmis olur.
    obs = env.reset(config=config, **(reset_kwargs or {}))
    losses: list = []
    info: dict = {}
    done = False

    while not done:
        acts = _act_all(agent, obs, env, train, shared=True)

        if health_probe is not None:
            for a in (AGENT_1, AGENT_2):
                if env.terminal(a) and not env.done:
                    health_probe.append(agent.q_value(a, obs[a], NOOP))
                    break

        state = env.state()
        next_obs, r_team, done, info = env.step(acts)
        next_state = env.state()

        if train:
            is_trunc = done and info.get("timeout", False)
            push_done = done and not is_trunc
            nm1 = env.physical_mask(AGENT_1)
            nm2 = env.physical_mask(AGENT_2)
            agent.push(obs[AGENT_1], acts[AGENT_1], obs[AGENT_2], acts[AGENT_2],
                       r_team, next_obs[AGENT_1], next_obs[AGENT_2],
                       push_done, nm1, nm2, state, next_state)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        obs = next_obs

    if train:
        agent.end_episode()          # n-adim penceresini bosalt (nstep.py)

    return info, losses
