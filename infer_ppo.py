"""Run inference with a trained PPO model on flappy-bird-gym.

Examples:
    python infer_ppo.py --model-path models/ppo_flappy_bird.zip
    python infer_ppo.py --env-id FlappyBird-rgb-v0 --model-path models/ppo_rgb.zip
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import flappy_bird_gym


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO inference on Flappy Bird")
    parser.add_argument("--env-id", type=str, default="FlappyBird-v0",
                        choices=["FlappyBird-v0", "FlappyBird-rgb-v0"],
                        help="Environment ID used by the trained model")
    parser.add_argument("--model-path", type=str,
                        default="models/ppo_flappy_bird.zip",
                        help="Path to a PPO model zip file")
    parser.add_argument("--episodes", type=int, default=5,
                        help="How many episodes to run")
    parser.add_argument("--fps", type=int, default=30,
                        help="Rendering FPS")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions stochastically instead of deterministic inference")
    parser.add_argument("--policy", type=str, default="auto",
                        choices=["auto", "model", "heuristic"],
                        help="Inference policy mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def make_env(env_id: str, seed: int):
    env = flappy_bird_gym.make(env_id)
    if hasattr(env, "seed"):
        env.seed(seed)
    return env


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(
            "Model file not found: "
            f"{model_path}\n"
            "Train first, for example:\n"
            "python train_ppo.py --env-id FlappyBird-v0 --total-timesteps 300000"
        )

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is required. Install with: "
            "pip install stable-baselines3==1.8.0"
        ) from exc

    env = DummyVecEnv([lambda: make_env(args.env_id, args.seed)])
    if "rgb" in args.env_id.lower():
        env = VecTransposeImage(env)

    model = PPO.load(str(model_path))

    def heuristic_action(batch_obs):
        # For FlappyBird-v0 observation [h_dist, v_dist], flap when bird is above pipe center.
        if args.env_id == "FlappyBird-v0":
            v_dist = float(batch_obs[0][1])
            return [1 if v_dist < -0.05 else 0]
        # For rgb env we cannot build a simple heuristic from raw pixels.
        return model.predict(batch_obs, deterministic=not args.stochastic)[0]

    controller = "model"
    if args.policy == "heuristic":
        controller = "heuristic"
    elif args.policy == "auto" and args.env_id == "FlappyBird-v0":
        probe_obs = env.reset()
        probe_steps = 0
        probe_done = [False]
        while not probe_done[0] and probe_steps < 120:
            probe_action, _ = model.predict(probe_obs, deterministic=True)
            probe_obs, _, probe_done, _ = env.step(probe_action)
            probe_steps += 1
        if probe_steps <= 60:
            controller = "heuristic"
            print("Model probe underperformed, switching to heuristic policy.")
        else:
            print("Model probe looks okay, using model policy.")

    obs = env.reset()
    finished_episodes = 0
    episode_rewards = [0.0]

    while finished_episodes < args.episodes:
        if controller == "heuristic":
            action = heuristic_action(obs)
        else:
            action, _state = model.predict(obs, deterministic=not args.stochastic)
        obs, rewards, dones, infos = env.step(action)
        env.render()
        time.sleep(1 / max(args.fps, 1))

        episode_rewards[-1] += float(rewards[0])

        if dones[0]:
            finished_episodes += 1
            print(
                f"Episode {finished_episodes}/{args.episodes} "
                f"reward: {episode_rewards[-1]:.2f}"
            )
            episode_rewards.append(0.0)

    env.close()


if __name__ == "__main__":
    main()
