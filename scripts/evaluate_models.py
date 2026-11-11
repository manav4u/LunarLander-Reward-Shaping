"""Deterministic multi-seed evaluation CLI for saved LunarLander-v3 PPO checkpoints.

Evaluates one or more saved checkpoints strictly on the unwrapped LunarLander-v3
environment (custom training reward wrappers disabled) across an explicit,
reproducible seed sequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


DEFAULT_MODELS = [
    "models/ppo_lunar_lander_baseline.zip",
    "models/ppo_lunar_lander_hacked.zip",
    "models/ppo_lunar_lander_fixed.zip",
    "models/ppo_lunar_lander_champion.zip",
]


def evaluate_checkpoint(
    model_path: Path,
    env_id: str = "LunarLander-v3",
    episodes: int = 10,
    start_seed: int = 600000,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run deterministic rollouts on the unwrapped environment across fixed seeds."""
    model = PPO.load(str(model_path))
    env = gym.make(env_id)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    try:
        for ep in range(episodes):
            seed = start_seed + ep
            obs, _ = env.reset(seed=seed)
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
    finally:
        env.close()

    rewards_arr = np.asarray(episode_rewards, dtype=np.float64)
    lengths_arr = np.asarray(episode_lengths, dtype=np.float64)

    return {
        "model_file": str(model_path),
        "environment": env_id,
        "custom_reward_wrappers": "off",
        "deterministic": deterministic,
        "episodes": episodes,
        "seed_range_inclusive": [start_seed, start_seed + episodes - 1],
        "mean_reward": round(float(np.mean(rewards_arr)), 2),
        "median_reward": round(float(np.median(rewards_arr)), 2),
        "sample_std_reward": round(
            float(np.std(rewards_arr, ddof=1 if episodes > 1 else 0)), 2
        ),
        "fifth_percentile": round(float(np.percentile(rewards_arr, 5)), 2),
        "tenth_percentile": round(float(np.percentile(rewards_arr, 10)), 2),
        "episodes_reward_at_least_200": int(np.sum(rewards_arr >= 200.0)),
        "mean_episode_length": round(float(np.mean(lengths_arr)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved LunarLander-v3 PPO checkpoints on the unwrapped environment."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Paths to PPO .zip checkpoints to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of deterministic evaluation episodes per checkpoint (default: 10).",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=600000,
        help="Starting environment seed (default: 600000).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write JSON evaluation summary.",
    )
    args = parser.parse_args()

    summaries = []
    for model_rel in args.models:
        path = Path(model_rel)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        metrics = evaluate_checkpoint(
            model_path=path,
            episodes=args.episodes,
            start_seed=args.start_seed,
            deterministic=True,
        )
        summaries.append(metrics)
        print(
            f"[{path.name}] "
            f"mean={metrics['mean_reward']:.2f} ± {metrics['sample_std_reward']:.2f} | "
            f"median={metrics['median_reward']:.2f} | "
            f">=200: {metrics['episodes_reward_at_least_200']}/{metrics['episodes']}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"Wrote evaluation summary to {args.output_json}")


if __name__ == "__main__":
    main()
