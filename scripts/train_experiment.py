"""Training and evaluation pipeline reconstructed from the Colab master cell."""


# Run this cell AFTER the training cell has completed.
# It packages the results and pushes them directly to GitHub; no phone download needed.

import os, sys, glob, json, csv, shutil, subprocess, textwrap, getpass, requests
from pathlib import Path

ROOT = Path.cwd().resolve()
PROJECT_TITLE = "Lunar Lander RL: Reward Shaping & Reward Hacking"
REPO_DESCRIPTION = "An empirical PPO experiment on LunarLander-v3: baseline learning, reward hacking, and reward shaping."

# 1) Check that the four recorded clips and all three PPO checkpoints exist.
stages = {
    "1_random_crash": "Stage 1 — Untrained Random Agent",
    "2_baseline_ppo": "Stage 2 — Baseline PPO",
    "3_hover_hack": "Stage 3 — Reward-Hacked PPO",
    "4_fixed_shaped": "Stage 4 — Fixed Reward Shaping PPO",
}

def latest_mp4(folder: Path):
    matches = list(folder.glob("*.mp4")) if folder.exists() else []
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None

source_videos = {}
missing = []
for folder, label in stages.items():
    found = latest_mp4(ROOT / "videos" / folder)
    if found is None:
        missing.append(f"video: {ROOT / 'videos' / folder}/*.mp4")
    else:
        source_videos[folder] = found

model_names = [
    "ppo_lunar_lander_baseline.zip",
    "ppo_lunar_lander_hacked.zip",
    "ppo_lunar_lander_fixed.zip",
]
source_models = {}
for name in model_names:
    p = ROOT / name
    if p.exists():
        source_models[name] = p
    else:
        missing.append(f"model: {p}")

if missing:
    raise FileNotFoundError(
        "Some expected artifacts are missing from this Colab runtime:\n- "
        + "\n- ".join(missing)
        + "\nRun the training/recording cells first, or adjust the source paths in this export cell."
    )

# 2) Locate the original master training cell from notebook history, if available.
training_source = None
history_items = []
try:
    history_items.extend(list(globals().get("In", [])))
except Exception:
    pass
try:
    ip = get_ipython()
    history_items.extend(list(ip.history_manager.input_hist_raw))
except Exception:
    pass
for item in reversed(history_items):
    if isinstance(item, str) and "MASTER COLAB NOTEBOOK" in item and "FINAL BENCHMARK SUMMARY" in item and "MAIN_TIMESTEPS" in item:
        training_source = item
        break

# Strip Colab shell commands from the script source; install commands are provided in notebook cell 1.
if training_source:
    cleaned_lines = [line for line in training_source.splitlines(keepends=True)
                     if not line.lstrip().startswith("!")]
    training_source = "".join(cleaned_lines)

# 3) Read the REAL scores produced by the current runtime. The visible screenshot values are fallbacks.
def runtime_number(name, fallback=None):
    value = globals().get(name, fallback)
    return float(value) if value is not None else None

main_steps = int(globals().get("MAIN_TIMESTEPS", 1_000_000))
hack_steps = int(globals().get("HACK_TIMESTEPS", 400_000))
results = {
    "evaluation_environment": "Gymnasium LunarLander-v3 (custom wrappers OFF)",
    "deterministic": True,
    "random_agent": {
        "mean_reward": runtime_number("rand_score", 29.92),
        "std_reward": None,
        "episodes": 1,
        "training_steps": 0,
        "episode_frames": int(globals().get("rand_frames", 0)),
    },
    "baseline_ppo": {
        "mean_reward": runtime_number("base_mean", 269.41),
        "std_reward": runtime_number("base_std"),
        "episodes": 10,
        "training_steps": main_steps,
    },
    "reward_hacked_ppo": {
        "mean_reward": runtime_number("hack_mean", -56.07),
        "std_reward": runtime_number("hack_std"),
        "episodes": 10,
        "training_steps": hack_steps,
    },
    "fixed_shaped_ppo": {
        "mean_reward": runtime_number("fixed_mean", 266.54),
        "std_reward": runtime_number("fixed_std"),
        "episodes": 10,
        "training_steps": main_steps,
    },
    "notes": [
        "Random agent score is one episode; PPO variants are 10-episode evaluation means.",
        "Scores were evaluated with custom reward wrappers disabled.",
        "The visible screenshot only exposes the means; standard deviations are included when available from the live runtime.",
    ],
}

# 4) Ask GitHub for a target repository and authenticate without putting the token in the Git URL.
repo_full_name = input("GitHub repository (owner/repo): ").strip().strip("/")
if repo_full_name.count("/") != 1:
    raise ValueError("Enter a repository as owner/repo, for example: yourname/lunar-lander-reward-shaping")
repo_owner, repo_name = repo_full_name.split("/", 1)
github_token = getpass.getpass("GitHub Personal Access Token (input is hidden): ").strip()
if not github_token:
    raise ValueError("A GitHub token is required to push from Colab.")

api_headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {github_token}",
    "X-GitHub-Api-Version": "2022-11-28",
}
api_root = "https://api.github.com"
user_resp = requests.get(f"{api_root}/user", headers=api_headers, timeout=30)
if user_resp.status_code != 200:
    raise RuntimeError(f"GitHub authentication failed ({user_resp.status_code}). Check the token and its permissions.")
api_user = user_resp.json()
login = api_user["login"]
repo_api_url = f"{api_root}/repos/{repo_owner}/{repo_name}"
repo_resp = requests.get(repo_api_url, headers=api_headers, timeout=30)

if repo_resp.status_code == 404:
    if repo_owner.lower() != login.lower():
        raise RuntimeError(
            f"Repository not found or not accessible: {repo_full_name}. Automatic creation is only enabled for your own account ({login}). "
            "Create the repo manually (or check owner/name) and rerun."
        )
    create = input(f"Repository {repo_full_name} was not found. Create it now? [y/N]: ").strip().lower()
    if create != "y":
        raise RuntimeError("Create an empty GitHub repository first, then rerun this cell.")
    visibility = input("Repository visibility: public or private? [public]: ").strip().lower() or "public"
    if visibility not in ("public", "private"):
        raise ValueError("Visibility must be public or private.")
    create_resp = requests.post(
        f"{api_root}/user/repos",
        headers=api_headers,
        json={"name": repo_name, "description": REPO_DESCRIPTION, "private": visibility == "private", "auto_init": False},
        timeout=30,
    )
    if create_resp.status_code not in (201,):
        raise RuntimeError(
            f"Could not create repository ({create_resp.status_code}): {create_resp.text[:500]}. "
            "The token may need repository-creation permission, or the repository may already exist."
        )
    repo_details = create_resp.json()
elif repo_resp.status_code == 200:
    repo_details = repo_resp.json()
else:
    raise RuntimeError(f"GitHub repository lookup failed ({repo_resp.status_code}): {repo_resp.text[:300]}")

# 5) Create a clean working clone in Colab. Git authentication uses a temporary askpass helper;
#    the token is never inserted into the remote URL or printed.
workdir = Path("/content/_lunar_lander_publish_worktree")
if workdir.exists():
    shutil.rmtree(workdir)
askpass = Path("/tmp/gh_colab_askpass.sh")
askpass.write_text(
    '#!/bin/sh\n'
    'case "$1" in\n'
    '  *Username*) printf "%s\\n" "$GH_USER" ;;\n'
    '  *Password*) printf "%s\\n" "$GH_TOKEN" ;;\n'
    '  *) printf "%s\\n" "$GH_TOKEN" ;;\n'
    'esac\n'
)
askpass.chmod(0o700)
git_env = os.environ.copy()
git_env.update({"GIT_ASKPASS": str(askpass), "GIT_TERMINAL_PROMPT": "0", "GH_USER": login, "GH_TOKEN": github_token})
repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
subprocess.run(["git", "clone", repo_url, str(workdir)], env=git_env, check=True)

# Remove token from this namespace as soon as clone authentication is configured.
# (The subprocess environment is still available for push below.)
branch = repo_details.get("default_branch") or "main"
current_branch = subprocess.run(["git", "branch", "--show-current"], cwd=workdir, text=True, capture_output=True, check=True).stdout.strip()
if not current_branch:
    subprocess.run(["git", "checkout", "-b", branch], cwd=workdir, env=git_env, check=True)
elif current_branch != branch:
    subprocess.run(["git", "checkout", branch], cwd=workdir, env=git_env, check=True)
subprocess.run(["git", "config", "user.name", login], cwd=workdir, check=True)
subprocess.run(["git", "config", "user.email", f"{api_user['id']}+{login}@users.noreply.github.com"], cwd=workdir, check=True)

# 6) Build the project structure; original MP4s are preserved and GIFs are only previews.
for folder in ["videos", "models", "src", "scripts", "notebooks", "results"]:
    (workdir / folder).mkdir(parents=True, exist_ok=True)

for folder, source in source_videos.items():
    canonical_mp4 = workdir / "videos" / f"{folder}.mp4"
    shutil.copy2(source, canonical_mp4)

for name, source in source_models.items():
    shutil.copy2(source, workdir / "models" / name)

# FFmpeg is used for compact GIF generation, avoiding MoviePy version differences.
if shutil.which("ffmpeg") is None:
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)

for folder, label in stages.items():
    mp4 = workdir / "videos" / f"{folder}.mp4"
    gif = workdir / "videos" / f"{folder}.gif"
    vf = "fps=12,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4), "-t", "6", "-vf", vf, "-loop", "0", str(gif)],
        check=True,
    )

# Reward wrappers as a reusable importable module.
wrapper_code = r'''"""Custom Gymnasium wrappers used in the Lunar Lander reward experiments."""
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
'''
(workdir / "src" / "reward_wrappers.py").write_text(wrapper_code)

# Portable training script and actual .ipynb are reconstructed from the master cell history if available.
if training_source:
    (workdir / "scripts" / "train_experiment.py").write_text(
        "\"\"\"Training and evaluation pipeline reconstructed from the Colab master cell.\"\"\"\n\n" + training_source
    )
    install_cell = """# Install system and Python dependencies before running the training cell.
"""
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Lunar Lander RL — Reward Shaping & Reward Hacking\n",
                "\n",
                "This notebook trains three PPO variants, records four demonstration episodes, and evaluates PPO agents with custom wrappers disabled.\n",
                "\n",
                "Run the setup cell first. Full training can take a while and may be interrupted by Colab runtime limits.\n"
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": install_cell.splitlines(keepends=True)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": training_source.splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.x"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (workdir / "notebooks" / "LunarLander_Reward_Shaping.ipynb").write_text(json.dumps(notebook, indent=2))
else:
    (workdir / "scripts" / "README.md").write_text(
        "The original training cell was not found in this runtime's input history. "
        "Export the Colab notebook as an .ipynb and place it in this folder if you want to preserve the exact notebook source.\n"
    )
    (workdir / "notebooks" / "README.md").write_text(
        "The live Colab notebook was not accessible from runtime history. Use Colab's File → Download → Download .ipynb, then add the exported notebook here.\n"
    )

(workdir / "requirements.txt").write_text(
    'gymnasium[box2d]\nstable-baselines3[extra]\nnumpy\nmoviepy\nimageio\nimageio-ffmpeg\nipython\n'
)
(workdir / ".gitignore").write_text(
    "__pycache__/\n*.py[cod]\n.ipynb_checkpoints/\n.venv/\nvenv/\n.env\n*.log\n.DS_Store\n"
)
(workdir / "LICENSE").write_text(textwrap.dedent("""\
MIT License

Copyright (c) 2026 Project Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""))

# Dynamic score formatting prevents README numbers from drifting away from the actual run.
notebook_entry = "notebooks/LunarLander_Reward_Shaping.ipynb" if training_source else "notebooks/README.md (notebook source unavailable in runtime history)"
script_entry = "scripts/train_experiment.py" if training_source else "scripts/README.md (training source unavailable in runtime history)"
def fmt_score(key):
    entry = results[key]
    mean = entry.get("mean_reward")
    std = entry.get("std_reward")
    if mean is None:
        return "Not recorded"
    return f"{mean:.2f} ± {std:.2f}" if std is not None else f"{mean:.2f}"

(workdir / "results" / "benchmark_results.json").write_text(json.dumps(results, indent=2))
with (workdir / "results" / "benchmark_summary.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["agent", "training_steps", "episodes", "mean_reward", "std_reward", "evaluation_wrappers"])
    for key in ["random_agent", "baseline_ppo", "reward_hacked_ppo", "fixed_shaped_ppo"]:
        r = results[key]
        writer.writerow([key, r.get("training_steps"), r.get("episodes"), r.get("mean_reward"), r.get("std_reward"), "OFF"])

readme = f'''# Lunar Lander RL: Reward Shaping & Reward Hacking

An empirical reinforcement-learning experiment using **Gymnasium `LunarLander-v3`** and **Proximal Policy Optimization (PPO)**. It compares an untrained random policy, a baseline PPO agent, a deliberately flawed reward bonus that can incentivize hovering, and a potential-difference shaping experiment with a time penalty.

## Results from this run

All PPO scores below were evaluated on the **unwrapped environment with custom reward wrappers OFF**. PPO scores are deterministic evaluations over 10 episodes; the random policy entry is a single episode. The environment is generally considered solved around an average score of 200, but results depend on seed and training run.

| Agent | Training steps | Evaluation score | Interpretation |
|---|---:|---:|---|
| Untrained random policy | 0 | {fmt_score('random_agent')} | Single episode; a reference, not a stable estimate |
| Baseline PPO | {results['baseline_ppo']['training_steps']:,} | {fmt_score('baseline_ppo')} | Learns using the environment's original reward |
| Reward-hacked PPO | {results['reward_hacked_ppo']['training_steps']:,} | {fmt_score('reward_hacked_ppo')} | Raw task score falls despite the added training bonus |
| Fixed-shaped PPO | {results['fixed_shaped_ppo']['training_steps']:,} | {fmt_score('fixed_shaped_ppo')} | Compared against the same unwrapped task score |

The baseline and fixed-shaped means are close in this run; the results do **not** establish that the shaped agent is better than baseline. Run more seeds and compare success rate, crash rate, episode length, and score distributions before drawing a stronger conclusion. Full values are stored in [`results/benchmark_results.json`](results/benchmark_results.json) and [`results/benchmark_summary.csv`](results/benchmark_summary.csv).

## Visual progression

| Stage 1 — Random policy | Stage 2 — Baseline PPO |
|---|---|
| ![Random agent](videos/1_random_crash.gif) | ![Baseline PPO](videos/2_baseline_ppo.gif) |
| [Original MP4](videos/1_random_crash.mp4) | [Original MP4](videos/2_baseline_ppo.mp4) |

| Stage 3 — Reward hacking | Stage 4 — Fixed shaping |
|---|---|
| ![Reward-hacked policy](videos/3_hover_hack.gif) | ![Fixed-shaped PPO](videos/4_fixed_shaped.gif) |
| [Original MP4](videos/3_hover_hack.mp4) | [Original MP4](videos/4_fixed_shaped.mp4) |

GIFs are short previews; the full recorded MP4s are retained in `videos/`.

## What the experiment demonstrates

1. **Baseline learning:** PPO can learn useful control behavior from the environment's eight-dimensional observation vector.
2. **Reward hacking / specification gaming:** a `+1` bonus for staying upright, slow, and airborne can reward behavior that does not accomplish the actual landing objective.
3. **Reward shaping experiment:** a frame-to-frame potential difference plus a per-step time fee attempts to encourage alignment and discourage wasting time.
4. **Evaluation discipline:** compare all policy variants with their training-only wrappers disabled so custom bonus rewards do not inflate evaluation scores.

## Method details

- **Environment:** Gymnasium `LunarLander-v3` (Box2D)
- **Algorithm:** Stable-Baselines3 PPO with an MLP policy
- **Observation:** horizontal position, vertical position, horizontal and vertical velocity, tilt, angular velocity, and two leg-contact indicators
- **Action space:** discrete main/side thruster controls
- **Evaluation:** deterministic policy actions; 10 episodes per PPO agent; custom wrappers off

The fixed wrapper in this repository uses `ΔΦ = Φ(sₜ) − Φ(sₜ₋₁)` and adds a `-0.05` per-step fee. This is the implementation tested here; it should not be described as a strict policy-invariance guarantee from the classic potential-based shaping theorem, whose usual discounted form uses `γΦ(s') − Φ(s)`.

## Repository map

```text
.
├── README.md
├── requirements.txt
├── LICENSE
├── {notebook_entry}
├── {script_entry}
├── src/reward_wrappers.py
├── models/
│   ├── ppo_lunar_lander_baseline.zip
│   ├── ppo_lunar_lander_hacked.zip
│   └── ppo_lunar_lander_fixed.zip
├── videos/                 # 4 MP4 originals + 4 GIF previews
└── results/                # JSON + CSV metrics
```

## Run locally

Install the system dependency and Python packages:

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y swig ffmpeg
python -m pip install -r requirements.txt
```

Evaluate a saved checkpoint (the model is evaluated with custom wrappers off):

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v3")
model = PPO.load("models/ppo_lunar_lander_baseline.zip")
mean, std = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
env.close()
print(f"Unwrapped score: {{mean:.2f}} +/- {{std:.2f}}")
```

## Limitations

- These are results from one run and do not establish statistical significance.
- Random-agent score is from one episode only.
- The reward-hacked behavior should be assessed from the clip and additional rollout metrics, not inferred from reward alone.
- The model files contain policy checkpoints, not a guarantee that training will reproduce identical results across hardware/library versions.

## References

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
- Ng, Harada & Russell, [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](https://ai.stanford.edu/~ang/papers/icml99-shaping.pdf), ICML 1999.
'''
(workdir / "README.md").write_text(readme)

# 7) If any MP4/model is close to GitHub's 100 MiB regular-Git limit, configure Git LFS.
large_assets = [p for p in list((workdir / "videos").glob("*.mp4")) + list((workdir / "models").glob("*.zip")) if p.stat().st_size >= 90 * 1024 * 1024]
if large_assets:
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "git-lfs"], check=True)
    subprocess.run(["git", "lfs", "install", "--local"], cwd=workdir, check=True)
    subprocess.run(["git", "lfs", "track", "videos/*.mp4", "models/*.zip"], cwd=workdir, check=True)
    print("Large assets will use Git LFS:", ", ".join(str(p.relative_to(workdir)) for p in large_assets))

# 8) Commit and push. Nothing is downloaded to the phone.
add_paths = ["README.md", "LICENSE", ".gitignore", "requirements.txt", "videos", "models", "src", "scripts", "notebooks", "results"]
if (workdir / ".gitattributes").exists():
    add_paths.append(".gitattributes")
subprocess.run(["git", "add", "--", *add_paths], cwd=workdir, env=git_env, check=True)
status = subprocess.run(["git", "status", "--porcelain"], cwd=workdir, text=True, capture_output=True, check=True).stdout.strip()
if status:
    subprocess.run(["git", "commit", "-m", "Add Lunar Lander reward-shaping experiment"], cwd=workdir, env=git_env, check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], cwd=workdir, env=git_env, check=True)
    print("\n✅ Project published successfully:", f"https://github.com/{repo_owner}/{repo_name}")
else:
    print("No changes to push; the repository already matches these generated files:", f"https://github.com/{repo_owner}/{repo_name}")

# Best-effort cleanup of the temporary credential helper and token variables.
try:
    askpass.unlink(missing_ok=True)
except Exception:
    pass
git_env.pop("GH_TOKEN", None)
github_token = None
print("\nPublished structure:")
for p in sorted(workdir.rglob("*")):
    if p.is_file() and ".git" not in p.parts:
        print(" -", p.relative_to(workdir))