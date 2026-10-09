"""MaskablePPO in NumPy (same algorithm as sb3-contrib ``MaskablePPO``).

Why NumPy: the team machines/CI and this container have no PyTorch; the
policy is small (about 100 inputs, 2 + S actions). The environment also
implements the sb3-contrib interface (``action_masks()``), so
``sb3_contrib.MaskablePPO`` can be dropped in where PyTorch is available.

Algorithm (Schulman et al. 2017; Huang & Ontanon 2022 invalid-action masking)
* actor and critic: separate tanh MLPs
* masked categorical: logits of infeasible actions are set to -1e9, so they
  get probability 0 and no gradient (masked entropy as well)
* GAE(lambda), advantage normalisation, clipped surrogate, value MSE,
  entropy bonus, global grad-norm clipping, Adam
* running observation normalisation (stored with the weights)

The saved policy records feature names, action count and the value
provider; ``load`` refuses a mismatching deployment.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

MASKED_LOGIT = -1e9
FORMAT = "pac_highlevel.maskable_ppo.v1"


class RunningNorm:
    def __init__(self, n):
        self.mean = np.zeros(n)
        self.var = np.ones(n)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, self.mean.size)
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_n
        self.mean = self.mean + delta * b_n / total
        m2 = self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / total
        self.var = m2 / total
        self.count = total

    def __call__(self, x):
        return np.clip((np.asarray(x, dtype=float) - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


class MLP:
    def __init__(self, sizes, rng, out_scale=1.0):
        self.W, self.b = [], []
        for i, (a, c) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = out_scale if i == len(sizes) - 2 else np.sqrt(2.0)
            # orthogonal init as in SB3
            q, _ = np.linalg.qr(rng.standard_normal((max(a, c), min(a, c))))
            w = q if a >= c else q.T
            self.W.append((gain * w[:a, :c]).astype(float))
            self.b.append(np.zeros(c))

    def forward(self, x):
        acts = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
            acts.append(h)
        return h, acts

    def backward(self, acts, grad_out):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        g = grad_out
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (1.0 - acts[i] ** 2)
        return gW, gb

    def params(self):
        return self.W + self.b

    def to_json(self):
        return {"W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b]}

    @classmethod
    def from_json(cls, data):
        m = cls.__new__(cls)
        m.W = [np.asarray(w, dtype=float) for w in data["W"]]
        m.b = [np.asarray(b, dtype=float) for b in data["b"]]
        return m


class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-5):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1**self.t)
            vh = v / (1 - self.b2**self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def masked_softmax(logits, mask):
    z = np.where(mask, logits, MASKED_LOGIT)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z) * mask
    p = e / e.sum(axis=-1, keepdims=True)
    return p


def _clip_grads(grads, max_norm):
    norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
    if norm > max_norm > 0:
        grads = [g * (max_norm / (norm + 1e-12)) for g in grads]
    return grads, norm


class MaskablePPO:
    def __init__(self, obs_dim, n_actions, config, *, feature_names=(), contract=None):
        self.cfg = config
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.feature_names = tuple(feature_names)
        self.contract = dict(contract or {})
        self.rng = np.random.default_rng(config.seed)
        hidden = tuple(config.hidden)
        self.pi = MLP((self.obs_dim, *hidden, self.n_actions), self.rng, out_scale=0.01)
        self.vf = MLP((self.obs_dim, *hidden, 1), self.rng, out_scale=1.0)
        self.pi_opt = Adam(self.pi.params(), config.learning_rate)
        self.vf_opt = Adam(self.vf.params(), config.learning_rate)
        self.norm = RunningNorm(self.obs_dim)
        self.history = []

    # ---------------------------------------------------------------
    def probs(self, obs, mask):
        x = self.norm(np.atleast_2d(obs))
        logits, _ = self.pi.forward(x)
        return masked_softmax(logits, np.atleast_2d(mask))

    def value(self, obs):
        v, _ = self.vf.forward(self.norm(np.atleast_2d(obs)))
        return v[:, 0]

    def act(self, obs, mask, deterministic=False):
        a, logp, v, _ = self.act_normed(obs, mask, deterministic)
        return a, logp, v

    def act_normed(self, obs, mask, deterministic=False):
        """Also returns the normalised observation used for the decision."""
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise ValueError("all actions masked")
        x = self.norm(np.atleast_2d(obs))
        logits, _ = self.pi.forward(x)
        p = masked_softmax(logits, mask[None, :])[0]
        a = int(np.argmax(p)) if deterministic else int(self.rng.choice(len(p), p=p))
        v, _ = self.vf.forward(x)
        return a, float(np.log(p[a] + 1e-12)), float(v[0, 0]), x[0]

    def get_weights(self):
        def arrays(m):
            return {"W": [w.copy() for w in m.W], "b": [b.copy() for b in m.b]}

        return {
            "pi": arrays(self.pi),
            "vf": arrays(self.vf),
            "norm": (self.norm.mean.copy(), self.norm.var.copy(), self.norm.count),
        }

    def set_weights(self, weights):
        self.pi = MLP.from_json(weights["pi"])
        self.vf = MLP.from_json(weights["vf"])
        self.norm.mean, self.norm.var, self.norm.count = weights["norm"]

    # ---------------------------------------------------------------
    def update(self, batch):
        """``batch["obs"]`` must already be normalised (as seen when acting)."""
        cfg = self.cfg
        obs = batch["obs"]
        masks = batch["masks"]
        actions = batch["actions"]
        old_logp = batch["logp"]
        adv = batch["adv"]
        ret = batch["ret"]
        n = len(actions)
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
        for _ in range(cfg.n_epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                x, m, a = obs[idx], masks[idx], actions[idx]
                A = adv[idx]
                A = (A - A.mean()) / (A.std() + 1e-8)
                k = len(idx)
                logits, acts = self.pi.forward(x)
                p = masked_softmax(logits, m)
                logp_all = np.where(m, np.log(np.maximum(np.where(m, p, 1.0), 1e-12)), 0.0)
                logp = logp_all[np.arange(k), a]
                ratio = np.exp(logp - old_logp[idx])
                unclipped = ratio * A
                clipped = np.clip(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * A
                active = unclipped <= clipped  # min() picks the unclipped term
                ent = -(p * logp_all).sum(1)
                # d(loss)/d(logp) for the surrogate
                g_logp = np.where(active, -ratio * A, 0.0) / k
                onehot = np.zeros_like(p)
                onehot[np.arange(k), a] = 1.0
                g_logits = g_logp[:, None] * (onehot - p)
                # entropy bonus: loss -= c * H ; dH/dz = -p (log p + H)
                g_logits += cfg.entropy_coef * (p * (logp_all + ent[:, None])) / k
                g_logits = np.where(m, g_logits, 0.0)
                gW, gb = self.pi.backward(acts, g_logits)
                grads, _ = _clip_grads(gW + gb, cfg.max_grad_norm)
                self.pi_opt.step(self.pi.params(), grads)

                v, vacts = self.vf.forward(x)
                diff = v[:, 0] - ret[idx]
                gv = (2.0 * cfg.value_coef * diff / k)[:, None]
                gW, gb = self.vf.backward(vacts, gv)
                grads, _ = _clip_grads(gW + gb, cfg.max_grad_norm)
                self.vf_opt.step(self.vf.params(), grads)

                stats["policy_loss"].append(float(-np.minimum(unclipped, clipped).mean()))
                stats["value_loss"].append(float((diff**2).mean()))
                stats["entropy"].append(float(ent.mean()))
                stats["approx_kl"].append(float(((ratio - 1) - np.log(ratio + 1e-12)).mean()))
                stats["clip_frac"].append(float((np.abs(ratio - 1) > cfg.clip_range).mean()))
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def imitate(self, obs, masks, actions, returns, epochs=20):
        """Behaviour cloning from a teacher (the rule policy) before PPO.

        ``obs`` must already be normalised. Cross-entropy on the masked
        policy and MSE of the critic on the teacher's discounted returns.
        """
        cfg = self.cfg
        n = len(actions)
        hist = []
        for _ in range(epochs):
            order = self.rng.permutation(n)
            losses, accs = [], []
            for start in range(0, n, cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                x, m, a, k = obs[idx], masks[idx], actions[idx], len(idx)
                logits, acts = self.pi.forward(x)
                p = masked_softmax(logits, m)
                onehot = np.zeros_like(p)
                onehot[np.arange(k), a] = 1.0
                g = np.where(m, (p - onehot) / k, 0.0)
                gW, gb = self.pi.backward(acts, g)
                grads, _ = _clip_grads(gW + gb, cfg.max_grad_norm)
                self.pi_opt.step(self.pi.params(), grads)
                v, vacts = self.vf.forward(x)
                diff = v[:, 0] - returns[idx]
                gW, gb = self.vf.backward(vacts, (2.0 * cfg.value_coef * diff / k)[:, None])
                grads, _ = _clip_grads(gW + gb, cfg.max_grad_norm)
                self.vf_opt.step(self.vf.params(), grads)
                losses.append(float(-np.log(p[np.arange(k), a] + 1e-12).mean()))
                accs.append(float((p.argmax(1) == a).mean()))
            hist.append({"bc_loss": float(np.mean(losses)), "bc_accuracy": float(np.mean(accs))})
        return hist

    # ---------------------------------------------------------------
    def save(self, path, extra=None):
        data = {
            "format": FORMAT,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "feature_names": list(self.feature_names),
            "contract": self.contract,
            "ppo_config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self.cfg).items()},
            "norm": {"mean": self.norm.mean.tolist(), "var": self.norm.var.tolist(), "count": self.norm.count},
            "pi": self.pi.to_json(),
            "vf": self.vf.to_json(),
            "history": self.history,
            **(extra or {}),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path, *, feature_names=None, contract=None):
        from .config import PPOConfig

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError("not a pac_highlevel MaskablePPO file")
        if feature_names is not None and tuple(data["feature_names"]) != tuple(feature_names):
            raise ValueError("feature layout differs from the trained policy")
        if contract is not None:
            for key, value in contract.items():
                if data["contract"].get(key) != value:
                    raise ValueError(
                        f"policy contract mismatch on {key}: trained {data['contract'].get(key)!r}, "
                        f"deployed {value!r}"
                    )
        cfg = PPOConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in data["ppo_config"].items()})
        agent = cls.__new__(cls)
        agent.cfg = cfg
        agent.obs_dim = data["obs_dim"]
        agent.n_actions = data["n_actions"]
        agent.feature_names = tuple(data["feature_names"])
        agent.contract = data["contract"]
        agent.rng = np.random.default_rng(cfg.seed)
        agent.pi = MLP.from_json(data["pi"])
        agent.vf = MLP.from_json(data["vf"])
        agent.pi_opt = Adam(agent.pi.params(), cfg.learning_rate)
        agent.vf_opt = Adam(agent.vf.params(), cfg.learning_rate)
        agent.norm = RunningNorm(agent.obs_dim)
        agent.norm.mean = np.asarray(data["norm"]["mean"])
        agent.norm.var = np.asarray(data["norm"]["var"])
        agent.norm.count = data["norm"]["count"]
        agent.history = data.get("history", [])
        return agent


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """dones[t] = True if the episode ended after step t."""
    n = len(rewards)
    adv = np.zeros(n)
    gae = 0.0
    for t in reversed(range(n)):
        next_v = last_value if t == n - 1 else values[t + 1]
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + np.asarray(values)
