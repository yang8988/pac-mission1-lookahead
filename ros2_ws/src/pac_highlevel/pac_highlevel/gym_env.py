"""Gymnasium wrapper (sb3-contrib ``MaskablePPO`` compatible).

```python
from sb3_contrib import MaskablePPO          # where PyTorch is available
env = HighLevelGymEnv(make_world)
model = MaskablePPO("MlpPolicy", env).learn(200_000)
```

``action_masks()`` is the hook sb3-contrib's ``ActionMasker`` /
``MaskablePPO`` look for. The same world, features and masks are used by the
NumPy trainer in this package.

Optional training aids (deployment never uses them):

- ``baseline(world) -> float``: return of a reference policy (the Rule) on a
  fresh copy of the same episode. It is subtracted at the last step, so the
  episode return becomes "better than the Rule on this box stream" and the
  large scenario-to-scenario difficulty spread drops out of the signal.
- ``teacher(world) -> HighLevelAction``: its choice for the state before each
  step is reported as ``info["teacher_action"]`` (behaviour-cloning labels on
  the states the learner actually visits).
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym = None

from .actions import action_count, from_index, to_index
from .features import feature_names, observe

if gym is not None:

    class HighLevelGymEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, make_world, slots, baseline=None, teacher=None):
            super().__init__()
            self.make_world = make_world
            self.slots = slots
            self.baseline = baseline
            self.teacher = teacher
            self.baseline_return = 0.0
            self.raw_return = 0.0
            n = len(feature_names(slots))
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
            self.action_space = spaces.Discrete(action_count(slots))
            self.world = None
            self.episode = -1

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            for _ in range(100):  # skip streams with nothing to decide (all NG)
                self.episode += 1
                self.world = self.make_world(self.episode)
                if not self.world.done:
                    break
            else:
                raise RuntimeError("100 consecutive episodes had nothing to decide")
            self.raw_return = 0.0
            if self.baseline is not None:
                self.baseline_return = float(self.baseline(self.make_world(self.episode)))
            return observe(self.world), {}

        def step(self, action):
            info = {}
            if self.teacher is not None:
                info["teacher_action"] = to_index(self.teacher(self.world))
            reward = self.world.step(from_index(action, self.slots))
            self.raw_return += reward
            done = self.world.done
            if done:
                info["summary"] = {**self.world.summary(), "return": self.raw_return}
                if self.baseline is not None:
                    reward -= self.baseline_return
                    info["summary"]["baseline_return"] = self.baseline_return
            return observe(self.world), float(reward), done, False, info

        def action_masks(self):
            return self.world.action_mask()

else:  # pragma: no cover
    HighLevelGymEnv = None
