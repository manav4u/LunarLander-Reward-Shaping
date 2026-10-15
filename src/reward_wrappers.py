"""Custom Gymnasium wrappers used in the Lunar Lander reward experiments."""
import numpy as np
import gymnasium as gym

class HoverHackWrapper(gym.Wrapper):
    """Adds +1 per airborne frame while upright and slow; this can incentivize hovering."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        vx, vy, tilt = obs[2], obs[3], obs[4]
        legs_down = obs[6] + obs[7]
        speed = np.sqrt(vx**2 + vy**2)
        if abs(tilt) < 0.25 and speed < 0.35 and legs_down == 0:
            reward += 1.0
        return obs, reward, terminated, truncated, info

class FixedRewardShapingWrapper(gym.Wrapper):
    """Adds frame-to-frame potential change and a per-step time fee."""
    def __init__(self, env, time_fee=-0.05):
        super().__init__(env)
        self.time_fee = time_fee
        self.prev_potential = None

    def _potential(self, obs):
        x, y, vx, vy, tilt, ang_vel = obs[:6]
        return -15.0 * abs(tilt) - 10.0 * abs(x) - 5.0 * abs(ang_vel)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_potential = self._potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_potential = self._potential(obs)
        improvement = curr_potential - self.prev_potential
        self.prev_potential = curr_potential
        return obs, reward + improvement + self.time_fee, terminated, truncated, info
