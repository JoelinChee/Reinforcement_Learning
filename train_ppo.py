"""Train a PPO agent on flappy-bird-gym environments.

Examples:
    python train_ppo.py --env-id FlappyBird-v0 --total-timesteps 300000
    python train_ppo.py --env-id FlappyBird-rgb-v0 --total-timesteps 500000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

import flappy_bird_gym
import gym


def use_local_stable_baselines3() -> Path:
    local_source = Path(__file__).resolve().parent / "stable_baselines3"
    if not local_source.exists():
        raise SystemExit(f"Local stable_baselines3 source not found: {local_source}")
    source_path = str(local_source)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    return local_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Flappy Bird")
    parser.add_argument("--env-id", type=str, default="FlappyBird-v0",
                        choices=["FlappyBird-v0", "FlappyBird-rgb-v0"],
                        help="Environment ID to train on")
    parser.add_argument("--total-timesteps", type=int, default=300_000,
                        help="Number of environment steps for training")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel envs (DummyVecEnv)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate for PPO")
    parser.add_argument("--n-steps", type=int, default=512,
                        help="Rollout steps per environment")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=8,
                        help="Optimization epochs per rollout")
    parser.add_argument("--target-kl", type=float, default=0.03,
                        help="Early stop updates if KL exceeds this value")
    parser.add_argument("--ent-coef", type=float, default=0.02,
                        help="Entropy coefficient for exploration")
    parser.add_argument("--reward-shaping", action="store_true",
                        help="Enable reward shaping for FlappyBird-v0")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-path", type=str,
                        default="models/ppo_flappy_bird",
                        help="Path prefix for the saved model")
    parser.add_argument("--tensorboard-log", type=str,
                        default=None,
                        help="TensorBoard log directory (disabled if omitted)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device, e.g. cpu, cuda, auto")
    parser.add_argument("--progress-bar", action="store_true",
                        help="Enable training progress bar (requires tqdm and rich)")
    return parser.parse_args()


def make_env(env_id: str, seed: int, rank: int, reward_shaping: bool) -> Callable:
    class ShapedRewardWrapper(gym.Wrapper):
        def __init__(self, env: gym.Env):
            super().__init__(env)
            self._prev_score = 0

        def reset(self):
            self._prev_score = 0
            return self.env.reset()

        def step(self, action):
            obs, _reward, done, info = self.env.step(action)
            h_dist, v_dist = float(obs[0]), float(obs[1])
            score = int(info.get("score", 0))
            score_gain = score - self._prev_score
            self._prev_score = score

            # Keep the bird near the pipe center and strongly reward passing pipes.
            shaped = 0.2 + (1.0 - min(abs(v_dist), 1.0)) * 0.6
            shaped += score_gain * 15.0
            if done:
                shaped -= 8.0

            return obs, shaped, done, info

    def _init():
        env = flappy_bird_gym.make(env_id)
        if hasattr(env, "seed"):
            env.seed(seed + rank)
        if reward_shaping and env_id == "FlappyBird-v0":
            env = ShapedRewardWrapper(env)
        return env

    return _init


def main() -> None:
    args = parse_args()
    local_stable_baselines3 = use_local_stable_baselines3()

    try:
        import stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
        from stable_baselines3.common.vec_env import VecMonitor
    except ImportError as exc:
        raise SystemExit(
            "Local stable_baselines3 dependencies are missing. "
            f"Using source directory: {local_stable_baselines3}"
        ) from exc

    imported_from = Path(stable_baselines3.__file__).resolve()
    if not imported_from.is_relative_to(local_stable_baselines3.resolve()):
        raise SystemExit(
            "Expected to import stable_baselines3 from local source, got: "
            f"{imported_from}"
        )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if args.tensorboard_log:
        Path(args.tensorboard_log).mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(args.env_id, args.seed, i, args.reward_shaping)
        for i in range(args.n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    policy = "MlpPolicy"
    if "rgb" in args.env_id.lower():
        policy = "CnnPolicy"
        vec_env = VecTransposeImage(vec_env)

    model = PPO(
        policy,
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        target_kl=args.target_kl,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.tensorboard_log,
        device=args.device,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=args.progress_bar,
    )
    model.save(str(save_path))
    vec_env.close()

    # Quick deterministic sanity-check on the raw environment.
    eval_env = flappy_bird_gym.make(args.env_id)
    eval_lengths = []
    for _ in range(5):
        obs = eval_env.reset()
        done = False
        steps = 0
        while not done and steps < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env.step(int(action))
            steps += 1
        eval_lengths.append(steps)
    eval_env.close()

    print(f"Training finished. Model saved to: {save_path}.zip")
    print(f"Quick eval episode lengths: {eval_lengths}")


if __name__ == "__main__":
    main()
