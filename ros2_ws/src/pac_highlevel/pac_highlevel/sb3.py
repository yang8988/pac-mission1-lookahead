"""PyTorch backend: sb3-contrib ``MaskablePPO`` on the same world and features.

Optional (``pip install torch sb3-contrib gymnasium``). The NumPy
implementation in ``ppo.py`` stays as the dependency-free fallback; both use
the identical world, observation, masks and rule-teacher warm start, so their
results are directly comparable.

Files: ``<name>.zip`` (sb3 model) + ``<name>.contract.json`` (feature layout,
value-provider contract and observation-normalisation statistics, checked on
load like the NumPy policy).

Observations are normalised with ``VecNormalize`` (running mean/std, clip 10),
like the NumPy learner's ``RunningNorm``. The statistics are seeded from the
rule-teacher data before imitation and stored with the policy, so evaluation
and deployment normalise exactly as training did.
"""

import json
from pathlib import Path

import numpy as np

from .actions import action_count, from_index
from .features import feature_names, observe
from .gym_env import HighLevelGymEnv

FORMAT = "pac_highlevel.sb3_maskable_ppo.v1"
CLIP_OBS = 10.0
NORM_EPS = 1e-8


def _require():
    try:
        import sb3_contrib  # noqa: F401
        import torch
    except ImportError as error:  # pragma: no cover - depends on the install
        raise RuntimeError("needs PyTorch: pip install torch sb3-contrib gymnasium") from error
    # tiny MLP: one thread is fastest and keeps forked workers safe
    torch.set_num_threads(1)


def make_vec_env(make_world, slots, n_envs=4, subprocess=True, normalize=True, baseline=None, teacher=None):
    """``n_envs`` environments; env k plays episodes k, k + n, k + 2n, ...

    With ``normalize`` the vector env is wrapped in ``VecNormalize``
    (observations only; rewards are already in pallet-volume units).
    ``baseline`` / ``teacher`` are passed to ``HighLevelGymEnv`` (training aids).
    """
    _require()
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    def env_fn(k):
        def make():
            return HighLevelGymEnv(lambda i: make_world(k + i * n_envs), slots, baseline=baseline, teacher=teacher)

        return make

    fns = [env_fn(k) for k in range(n_envs)]
    if subprocess and n_envs > 1:
        venv = SubprocVecEnv(fns, start_method="fork")
    else:
        venv = DummyVecEnv(fns)
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=CLIP_OBS, epsilon=NORM_EPS)
    return venv


def new_model(env, ppo_config, seed=None, verbose=0):
    _require()
    from sb3_contrib import MaskablePPO

    cfg = ppo_config
    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=max(1, cfg.n_steps // env.num_envs),
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.entropy_coef,
        vf_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs={"net_arch": {"pi": list(cfg.hidden), "vf": list(cfg.hidden)}},
        seed=cfg.seed if seed is None else seed,
        verbose=verbose,
        device="cpu",
    )


def imitate(model, data, epochs=20, batch_size=128, lr=1e-3):
    """Rule-teacher warm start: masked cross-entropy + value regression.

    With ``VecNormalize`` the running statistics are first updated with the
    teacher observations, and the policy learns on normalised observations.
    """
    _require()
    import torch

    raw = np.asarray(data["obs"], dtype=np.float32)
    vecnorm = model.get_vec_normalize_env()
    if vecnorm is not None:
        vecnorm.obs_rms.update(raw)
        raw = vecnorm.normalize_obs(raw).astype(np.float32)
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    obs = torch.as_tensor(raw, dtype=torch.float32)
    masks = data["masks"]
    actions = torch.as_tensor(data["actions"], dtype=torch.long)
    returns = torch.as_tensor(data["returns"], dtype=torch.float32)
    n = len(actions)
    rng = np.random.default_rng(0)
    hist = []
    for _ in range(epochs):
        order = rng.permutation(n)
        losses, accs = [], []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            dist = policy.get_distribution(obs[idx], action_masks=masks[idx])
            logp = dist.log_prob(actions[idx])
            values = policy.predict_values(obs[idx]).squeeze(-1)
            loss = -logp.mean() + 0.5 * torch.nn.functional.mse_loss(values, returns[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                probs = dist.distribution.probs
                accs.append(float((probs.argmax(-1) == actions[idx]).float().mean()))
            losses.append(float(-logp.mean().detach()))
        hist.append({"phase": "imitation", "bc_loss": float(np.mean(losses)),
                     "bc_accuracy": float(np.mean(accs))})
    return hist


def teacher_bc_callback(coef_start=0.5, coef_end=0.1, epochs=2, batch_size=128):
    """Keep the learner close to the teacher on the states it visits.

    After every PPO update, a few behaviour-cloning steps (masked
    cross-entropy towards ``info["teacher_action"]``) run on the previous
    rollout, weighted by a coefficient that decays linearly from
    ``coef_start`` to ``coef_end`` over training. PPO can still move away from
    the teacher where that pays, but drifting to worse-than-teacher choices
    costs it (DAPG / DAgger-style regularisation).
    """
    _require()
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    class TeacherBC(BaseCallback):
        def __init__(self):
            super().__init__()
            self.labels = []
            self.pending = None
            self.rng = np.random.default_rng(0)

        def _on_rollout_start(self):
            if self.pending is not None:
                self._clone(*self.pending)
                self.pending = None
            self.labels = []

        def _on_step(self):
            self.labels.append([info.get("teacher_action", -1) for info in self.locals["infos"]])
            return True

        def _on_rollout_end(self):
            buf = self.model.rollout_buffer
            labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
            obs = buf.observations.reshape(len(labels), -1)
            masks = buf.action_masks.reshape(len(labels), -1).astype(bool)
            keep = labels >= 0
            self.pending = (obs[keep].copy(), masks[keep].copy(), labels[keep])

        def _clone(self, obs, masks, labels):
            total = max(1, getattr(self.model, "_total_timesteps", 1) or 1)
            frac = min(1.0, self.num_timesteps / total)
            coef = coef_start + (coef_end - coef_start) * frac
            if coef <= 0 or len(labels) == 0:
                return
            policy = self.model.policy
            opt = policy.optimizer
            o = torch.as_tensor(obs, dtype=torch.float32)
            a = torch.as_tensor(labels, dtype=torch.long)
            losses, accs = [], []
            for _ in range(epochs):
                order = self.rng.permutation(len(a))
                for start in range(0, len(a), batch_size):
                    idx = order[start : start + batch_size]
                    dist = policy.get_distribution(o[idx], action_masks=masks[idx])
                    ce = -dist.log_prob(a[idx]).mean()
                    opt.zero_grad()
                    (coef * ce).backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.model.max_grad_norm)
                    opt.step()
                    with torch.no_grad():
                        accs.append(float((dist.distribution.probs.argmax(-1) == a[idx]).float().mean()))
                    losses.append(float(ce.detach()))
            self.logger.record("bc/coef", coef)
            self.logger.record("bc/loss", float(np.mean(losses)))
            self.logger.record("bc/agreement", float(np.mean(accs)))

    return TeacherBC()


def save(model, path, slots, contract, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path.with_suffix("")))
    meta = {"format": FORMAT, "feature_names": list(feature_names(slots)),
            "n_actions": action_count(slots), "contract": contract, **(extra or {})}
    vecnorm = model.get_vec_normalize_env()
    if vecnorm is not None:
        rms = vecnorm.obs_rms
        meta["obs_norm"] = {"mean": rms.mean.tolist(), "var": rms.var.tolist(), "count": float(rms.count),
                            "clip": float(vecnorm.clip_obs), "eps": float(vecnorm.epsilon)}
    path.with_suffix(".contract.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def load(path, *, slots, contract=None):
    _require()
    from sb3_contrib import MaskablePPO

    path = Path(path)
    meta = json.loads(path.with_suffix(".contract.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError("not a pac_highlevel sb3 policy")
    if tuple(meta["feature_names"]) != feature_names(slots):
        raise ValueError("feature layout differs from the trained policy")
    for key, value in (contract or {}).items():
        if meta["contract"].get(key) != value:
            raise ValueError(f"policy contract mismatch on {key}: trained "
                             f"{meta['contract'].get(key)!r}, deployed {value!r}")
    model = MaskablePPO.load(str(path.with_suffix("")), device="cpu")
    model.pac_obs_norm = meta.get("obs_norm")
    return model


def normalize_obs(obs, norm):
    """Same formula as ``VecNormalize.normalize_obs`` with frozen statistics."""
    if not norm:
        return np.asarray(obs, dtype=np.float32)
    mean = np.asarray(norm["mean"])
    var = np.asarray(norm["var"])
    x = (np.asarray(obs, dtype=float) - mean) / np.sqrt(var + norm["eps"])
    return np.clip(x, -norm["clip"], norm["clip"]).astype(np.float32)


def chooser(model, deterministic=True):
    norm = getattr(model, "pac_obs_norm", None)

    def choose(world):
        action, _ = model.predict(normalize_obs(observe(world), norm), action_masks=world.action_mask(),
                                  deterministic=deterministic)
        return from_index(int(action), world.slots)

    return choose
