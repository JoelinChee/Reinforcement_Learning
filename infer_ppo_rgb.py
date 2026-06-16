"""Run inference with a trained PPO model on the RGB Flappy Bird environment.

Examples:
    python infer_ppo_rgb.py --model-path models/ppo_flappy_bird_rgb.zip
    python infer_ppo_rgb.py --episodes 3 --render-mode none
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gym
import numpy as np

from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB


ENV_ID = "FlappyBird-rgb-v0"


class RGBObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=env.observation_space.shape,
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.astype(np.uint8, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO inference on RGB Flappy Bird")
    parser.add_argument("--model-path", type=str,
                        default="models/ppo_flappy_bird_rgb.zip",
                        help="Path to a PPO RGB model zip file")
    parser.add_argument("--episodes", type=int, default=5,
                        help="How many episodes to run")
    parser.add_argument("--fps", type=int, default=30,
                        help="Rendering FPS")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Optional maximum steps per episode")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions stochastically instead of deterministic inference")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--screen-width", type=int, default=288,
                        help="RGB environment screen width used during training")
    parser.add_argument("--screen-height", type=int, default=512,
                        help="RGB environment screen height used during training")
    parser.add_argument("--render-mode", type=str, default="human",
                        choices=["human", "rgb_array", "none"],
                        help="How to render during inference")
    return parser.parse_args()


def make_env(seed: int, screen_size: tuple[int, int]) -> gym.Env:
    env = FlappyBirdEnvRGB(screen_size=screen_size)
    if hasattr(env, "seed"):
        env.seed(seed)
    return RGBObservationWrapper(env)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(
            "Model file not found: "
            f"{model_path}\n"
            "Train first, for example:\n"
            "python train_ppo_rgb.py --total-timesteps 500000"
        )

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is required. Install with: "
            "pip install stable-baselines3==1.8.0"
        ) from exc

    screen_size = (args.screen_width, args.screen_height)
    env = DummyVecEnv([lambda: make_env(args.seed, screen_size)])
    env = VecTransposeImage(env)
    model = PPO.load(str(model_path))

    obs = env.reset()
    finished_episodes = 0
    episode_rewards = [0.0]
    episode_steps = 0
    completed_rewards = []
    completed_scores = []
    completed_steps = []

    while finished_episodes < args.episodes:
        action, _state = model.predict(obs, deterministic=not args.stochastic)
        obs, rewards, dones, infos = env.step(action)
        episode_steps += 1

        if args.render_mode != "none":
            env.render(mode=args.render_mode)
            time.sleep(1 / max(args.fps, 1))

        episode_rewards[-1] += float(rewards[0])

        reached_max_steps = args.max_steps is not None and episode_steps >= args.max_steps
        if dones[0] or reached_max_steps:
            finished_episodes += 1
            score = infos[0].get("score", 0)
            print(
                f"Episode {finished_episodes}/{args.episodes} "
                f"reward: {episode_rewards[-1]:.2f} "
                f"score: {score} steps: {episode_steps}"
            )
            completed_rewards.append(episode_rewards[-1])
            completed_scores.append(float(score))
            completed_steps.append(float(episode_steps))
            episode_rewards.append(0.0)
            episode_steps = 0
            if reached_max_steps and not dones[0]:
                obs = env.reset()

    env.close()
    print(
        "Summary "
        f"episodes={len(completed_rewards)} "
        f"mean_reward={np.mean(completed_rewards):.2f} "
        f"mean_score={np.mean(completed_scores):.2f} "
        f"mean_steps={np.mean(completed_steps):.2f} "
        f"max_score={np.max(completed_scores):.0f} "
        f"max_steps={np.max(completed_steps):.0f}"
    )
    print(f"Inference finished on {ENV_ID}.")


if __name__ == "__main__":
    main()