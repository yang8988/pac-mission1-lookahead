"""Training / evaluation loops shared by scripts and tests.

``make_world(i)`` returns a fresh :class:`PalletizingWorld` for episode
``i`` (the caller decides the scenario split and seeds). Rollouts can be
collected by worker processes (``workers > 1``); every worker samples with
the current policy weights and returns the *normalised* observations it
acted on, so the PPO ratio starts at exactly 1.
"""

import multiprocessing as mp
import time

import numpy as np

from .actions import action_count, from_index, to_index
from .features import feature_names, observe
from .ppo import MaskablePPO, compute_gae

KEYS = ("obs", "raw", "masks", "actions", "logp", "values", "rewards", "dones")


def policy_contract(config, cand_config_name="candidates.yaml"):
    return {
        "slots": config.buffer.slots,
        "value_provider": config.features.value_provider,
        "placer": "dblf",
        "candidate_config": cand_config_name,
    }


def new_agent(config, contract=None):
    names = feature_names(config.buffer.slots)
    return MaskablePPO(
        len(names),
        action_count(config.buffer.slots),
        config.ppo,
        feature_names=names,
        contract=contract or policy_contract(config),
    )


def run_policy(world, choose):
    """Roll one episode with ``choose(world) -> HighLevelAction``."""
    total = 0.0
    while not world.done:
        total += world.step(choose(world))
    out = world.summary()
    out["return"] = total
    return out


def agent_chooser(agent, deterministic=True):
    def choose(world):
        a, _, _ = agent.act(observe(world), world.action_mask(), deterministic=deterministic)
        return from_index(a, world.slots)

    return choose


class _Collector:
    """Steps one stream of episodes; episode ids are ``offset + k * stride``."""

    def __init__(self, make_world, offset, stride):
        self.make_world = make_world
        self.offset = offset
        self.stride = stride
        self.k = 0
        self.world = self._new()
        self.ep_return = 0.0

    def _new(self):
        while True:
            world = self.make_world(self.offset + self.k * self.stride)
            self.k += 1
            if not world.done:
                return world

    def collect(self, agent, n):
        traj = {k: [] for k in KEYS}
        segments = []
        finished = []
        start = 0
        for _ in range(n):
            raw = observe(self.world)
            mask = self.world.action_mask()
            a, logp, v, x = agent.act_normed(raw, mask)
            r = self.world.step(from_index(a, self.world.slots))
            self.ep_return += r
            for key, val in zip(KEYS, (x, raw, mask, a, logp, v, r, self.world.done)):
                traj[key].append(val)
            if self.world.done:
                summary = self.world.summary()
                summary["return"] = self.ep_return
                summary.pop("pallets", None)
                finished.append(summary)
                segments.append((start, len(traj["actions"]), 0.0))
                start = len(traj["actions"])
                self.world = self._new()
                self.ep_return = 0.0
        if start < len(traj["actions"]):
            _, _, last, _ = agent.act_normed(observe(self.world), self.world.action_mask())
            segments.append((start, len(traj["actions"]), last))
        return traj, segments, finished


def _worker(conn, make_world, offset, stride, agent):
    collector = _Collector(make_world, offset, stride)
    agent.rng = np.random.default_rng(agent.cfg.seed * 1000 + offset + 1)
    while True:
        msg = conn.recv()
        if msg is None:
            return
        weights, n = msg
        agent.set_weights(weights)
        conn.send(collector.collect(agent, n))


class ParallelCollector:
    def __init__(self, make_world, agent, workers):
        ctx = mp.get_context("fork")
        self.conns = []
        self.procs = []
        for w in range(workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, make_world, w, workers, agent), daemon=True)
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)

    def collect(self, agent, n):
        per = -(-n // len(self.conns))
        weights = agent.get_weights()
        for c in self.conns:
            c.send((weights, per))
        return [c.recv() for c in self.conns]

    def close(self):
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(timeout=5)


def _merge(parts, cfg):
    traj = {k: [] for k in KEYS}
    adv, ret = [], []
    finished = []
    for t, segments, fin in parts:
        finished += fin
        for s, e, last in segments:
            a_, r_ = compute_gae(
                t["rewards"][s:e], t["values"][s:e], t["dones"][s:e], last, cfg.gamma, cfg.gae_lambda
            )
            adv.append(a_)
            ret.append(r_)
        for k in KEYS:
            traj[k] += t[k]
    batch = {
        "obs": np.asarray(traj["obs"], dtype=float),
        "raw": np.asarray(traj["raw"], dtype=float),
        "masks": np.asarray(traj["masks"], dtype=bool),
        "actions": np.asarray(traj["actions"], dtype=int),
        "logp": np.asarray(traj["logp"], dtype=float),
        "adv": np.concatenate(adv) if adv else np.zeros(0),
        "ret": np.concatenate(ret) if ret else np.zeros(0),
    }
    return batch, finished


_TEACHER = None  # (make_world, policy, gamma); set before forking workers


def _teacher_episode(i):
    make_world, policy, gamma, relative = _TEACHER
    world = make_world(i)
    obs, masks, actions, rewards = [], [], [], []
    while not world.done:
        obs.append(observe(world))
        masks.append(world.action_mask())
        action = policy(world)
        actions.append(to_index(action))
        rewards.append(world.step(action))
    if relative and rewards:  # teacher measured against itself: episode return 0
        rewards[-1] -= sum(rewards)
    returns = np.zeros(len(rewards))
    acc = 0.0
    for t in reversed(range(len(rewards))):
        acc = rewards[t] + gamma * acc
        returns[t] = acc
    return obs, masks, actions, returns


def collect_teacher(make_world, policy, episodes, *, gamma, workers=1, first_episode=10**6, relative=False):
    """Teacher roll-outs: raw observations, masks, actions, discounted returns.

    ``relative`` matches the rule-baseline reward of ``HighLevelGymEnv``: the
    teacher's own episode return is subtracted at its last step.
    """
    global _TEACHER
    _TEACHER = (make_world, policy, gamma, relative)
    ids = [first_episode + i for i in range(episodes)]
    try:
        if workers > 1:
            with mp.get_context("fork").Pool(workers) as pool:
                parts = pool.map(_teacher_episode, ids, chunksize=1)
        else:
            parts = [_teacher_episode(i) for i in ids]
    finally:
        _TEACHER = None
    return {
        "obs": np.asarray([o for p in parts for o in p[0]], dtype=np.float32),
        "masks": np.asarray([m for p in parts for m in p[1]], dtype=bool),
        "actions": np.asarray([a for p in parts for a in p[2]], dtype=int),
        "returns": np.concatenate([p[3] for p in parts]),
    }


def imitate_teacher(agent, make_world, policy, episodes, *, workers=1, epochs=20, first_episode=10**6):
    """Warm-start the agent from a teacher policy (episode ids are offset so
    they do not repeat the PPO episodes)."""
    data = collect_teacher(make_world, policy, episodes, gamma=agent.cfg.gamma, workers=workers,
                           first_episode=first_episode)
    raw = data["obs"].astype(float)
    agent.norm.update(raw)
    hist = agent.imitate(agent.norm(raw), data["masks"], data["actions"], data["returns"], epochs=epochs)
    for row in hist:
        row["phase"] = "imitation"
    agent.history.extend(hist)
    return hist


def train(agent, make_world, total_steps, *, workers=1, log=print, eval_fn=None, eval_every=0):
    cfg = agent.cfg
    collector = ParallelCollector(make_world, agent, workers) if workers > 1 else _Collector(make_world, 0, 1)
    started = time.perf_counter()
    steps = 0
    update = 0
    finished = []
    try:
        while steps < total_steps:
            if workers > 1:
                parts = collector.collect(agent, cfg.n_steps)
            else:
                parts = [collector.collect(agent, cfg.n_steps)]
            batch, fin = _merge(parts, cfg)
            finished += fin
            steps += len(batch["actions"])
            stats = agent.update(batch)
            agent.norm.update(batch["raw"])  # affects the next rollout only
            update += 1
            recent = finished[-20:]
            row = {
                "update": update,
                "steps": steps,
                "episodes": len(finished),
                "elapsed_s": round(time.perf_counter() - started, 1),
                **{k: round(v, 5) for k, v in stats.items()},
                "recent_return": float(np.mean([f["return"] for f in recent])) if recent else None,
                "recent_pallet_eq": float(np.mean([f["pallet_equivalents"] for f in recent])) if recent else None,
                "recent_fill": float(np.mean([f["fill_per_pallet_used"] for f in recent])) if recent else None,
            }
            if eval_fn is not None and eval_every and update % eval_every == 0:
                row["eval"] = eval_fn(agent)
            agent.history.append(row)
            log(row)
    finally:
        if workers > 1:
            collector.close()
    return finished


__all__ = [
    "ParallelCollector",
    "agent_chooser",
    "collect_teacher",
    "imitate_teacher",
    "new_agent",
    "policy_contract",
    "run_policy",
    "train",
]
