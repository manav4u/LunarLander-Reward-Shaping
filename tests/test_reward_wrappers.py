"""Unit tests for custom LunarLander reward wrappers (src/reward_wrappers.py)."""

import math
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Use installed numpy/gymnasium when present; provide minimal compatible stubs
# when running in a bare environment without third-party packages installed.
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np_stub = types.ModuleType("numpy")
    np_stub.sqrt = math.sqrt
    np_stub.float32 = float
    np_stub.inf = float("inf")
    np_stub.zeros = lambda n, dtype=None: [0.0] * n
    np_stub.array = lambda seq, dtype=None: [float(x) for x in seq]
    np_stub.asarray = lambda seq, dtype=None: [float(x) for x in seq]
    sys.modules["numpy"] = np_stub
    import numpy as np  # type: ignore

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym_stub = types.ModuleType("gymnasium")

    class _Env:
        def reset(self, *, seed=None, options=None):
            return None, {}

    class _Wrapper:
        def __init__(self, env):
            self.env = env

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def step(self, action):
            return self.env.step(action)

    gym_stub.Env = _Env
    gym_stub.Wrapper = _Wrapper
    sys.modules["gymnasium"] = gym_stub
    import gymnasium as gym  # type: ignore

from src.reward_wrappers import FixedRewardShapingWrapper, HoverHackWrapper


class DummyLunarEnv(gym.Env):
    """Minimal deterministic 8-D observation environment for testing reward wrappers."""

    def __init__(self, initial_obs, step_sequence):
        super().__init__()
        self._initial_obs = [float(x) for x in initial_obs]
        self._steps = [([float(x) for x in obs], float(r)) for obs, r in step_sequence]
        self._idx = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        return list(self._initial_obs), {}

    def step(self, action):
        obs, reward = self._steps[self._idx]
        self._idx += 1
        terminated = self._idx >= len(self._steps)
        return list(obs), reward, terminated, False, {}


class TestRewardWrappers(unittest.TestCase):
    def test_hover_hack_adds_bonus_only_when_airborne_upright_and_slow(self):
        # obs: [x, y, vx, vy, tilt, ang_vel, leg1, leg2]
        init_obs = [0.0] * 8
        steps = [
            # Airborne (legs=0), upright (|tilt|=0.1 < 0.25), slow (speed=0.1 < 0.35) -> +1.0 bonus
            ([0.0, 1.0, 0.06, 0.08, 0.10, 0.0, 0.0, 0.0], 0.5),
            # Landed (leg1=1.0) -> no bonus
            ([0.0, 0.0, 0.00, 0.00, 0.00, 0.0, 1.0, 0.0], 0.5),
            # High tilt (tilt=0.4 >= 0.25) -> no bonus
            ([0.0, 1.0, 0.00, 0.00, 0.40, 0.0, 0.0, 0.0], 0.5),
            # High speed (vx=0.4 >= 0.35) -> no bonus
            ([0.0, 1.0, 0.40, 0.00, 0.00, 0.0, 0.0, 0.0], 0.5),
        ]
        env = HoverHackWrapper(DummyLunarEnv(init_obs, steps))
        env.reset()

        _, r1, _, _, _ = env.step(0)
        self.assertAlmostEqual(r1, 1.5, places=6)

        _, r2, _, _, _ = env.step(0)
        self.assertAlmostEqual(r2, 0.5, places=6)

        _, r3, _, _, _ = env.step(0)
        self.assertAlmostEqual(r3, 0.5, places=6)

        _, r4, _, _, _ = env.step(0)
        self.assertAlmostEqual(r4, 0.5, places=6)

    def test_fixed_reward_shaping_potential_difference_and_time_fee(self):
        # Potential: Φ(s) = -15*|tilt| - 10*|x| - 5*|ang_vel|
        # Initial state: x=0.2, tilt=0.1, ang_vel=0.2 -> Φ_0 = -1.5 - 2.0 - 1.0 = -4.5
        init_obs = [0.2, 1.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0]
        # Step 1 state: perfectly aligned (x=0, tilt=0, ang_vel=0) -> Φ_1 = 0.0
        # Improvement: ΔΦ = 0.0 - (-4.5) = +4.5; with time_fee = -0.05 -> +4.45
        step1_obs = [0.0] * 8

        env = FixedRewardShapingWrapper(DummyLunarEnv(init_obs, [(step1_obs, 1.0)]), time_fee=-0.05)
        env.reset()
        _, shaped_reward, _, _, _ = env.step(0)

        self.assertAlmostEqual(shaped_reward, 1.0 + 4.5 - 0.05, places=5)


if __name__ == "__main__":
    unittest.main()
