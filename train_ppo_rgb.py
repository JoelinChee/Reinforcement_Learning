"""Train a PPO agent on the RGB Flappy Bird environment.

Examples:
    python train_ppo_rgb.py --total-timesteps 500000
    python train_ppo_rgb.py --save-path models/ppo_flappy_bird_rgb
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

import gym
import numpy as np

from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB
from flappy_bird_gym.envs.game_logic import PIPE_HEIGHT, PIPE_WIDTH
from flappy_bird_gym.envs.game_logic import PLAYER_HEIGHT, PLAYER_WIDTH


ENV_ID = "FlappyBird-rgb-v0"


def use_local_stable_baselines3() -> Path:
    local_source = Path(__file__).resolve().parent / "stable_baselines3"
    if not local_source.exists():
        raise SystemExit(f"Local stable_baselines3 source not found: {local_source}")
    source_path = str(local_source)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    return local_source


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


class RGBShapedRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_score = 0

    def reset(self):
        self._prev_score = 0
        return self.env.reset()

    def step(self, action):
        obs, _reward, done, info = self.env.step(action)
        game = self.env.unwrapped._game

        next_upper_pipe = game.upper_pipes[0]
        next_lower_pipe = game.lower_pipes[0]
        for upper_pipe, lower_pipe in zip(game.upper_pipes, game.lower_pipes):
            pipe_mid_x = upper_pipe["x"] + PIPE_WIDTH / 2
            player_mid_x = game.player_x + PLAYER_WIDTH / 2
            if pipe_mid_x >= player_mid_x:
                next_upper_pipe = upper_pipe
                next_lower_pipe = lower_pipe
                break

        gap_center_y = (
            next_upper_pipe["y"] + PIPE_HEIGHT + next_lower_pipe["y"]
        ) / 2
        player_center_y = game.player_y + PLAYER_HEIGHT / 2
        normalized_v_dist = abs(gap_center_y - player_center_y) / game._screen_height

        score = int(info.get("score", 0))
        score_gain = score - self._prev_score
        self._prev_score = score

        shaped_reward = 0.1 + max(0.0, 1.0 - normalized_v_dist) * 0.4
        shaped_reward += score_gain * 20.0
        if done:
            shaped_reward -= 10.0

        return obs, shaped_reward, done, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on RGB Flappy Bird")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
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
                        help="Enable RGB reward shaping during training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--screen-width", type=int, default=288,
                        help="RGB environment screen width")
    parser.add_argument("--screen-height", type=int, default=512,
                        help="RGB environment screen height")
    parser.add_argument("--save-path", type=str,
                        default="models/ppo_flappy_bird_rgb",
                        help="Path prefix for the saved RGB model")
    parser.add_argument("--checkpoint-freq", type=int, default=0,
                        help="Save a checkpoint every N environment steps; disabled if 0")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints_rgb",
                        help="Directory for periodic checkpoints")
    parser.add_argument("--tensorboard-log", type=str,
                        default=None,
                        help="TensorBoard log directory (disabled if omitted)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device, e.g. cpu, cuda, auto")
    parser.add_argument("--progress-bar", action="store_true",
                        help="Enable training progress bar (requires tqdm and rich)")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="How many quick eval episodes to run after training")
    parser.add_argument("--eval-max-steps", type=int, default=1000,
                        help="Maximum steps per quick eval episode")
    return parser.parse_args()


def make_env(
    seed: int,
    rank: int,
    screen_size: tuple[int, int],
    reward_shaping: bool,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = FlappyBirdEnvRGB(screen_size=screen_size)
        if hasattr(env, "seed"):
            env.seed(seed + rank)
        if reward_shaping:
            env = RGBShapedRewardWrapper(env)
        return RGBObservationWrapper(env)

    return _init


def main() -> None:
    args = parse_args()
    local_stable_baselines3 = use_local_stable_baselines3()

    try:
        import stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        from stable_baselines3.common.vec_env import VecTransposeImage
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

    screen_size = (args.screen_width, args.screen_height)
    env_fns = [
        make_env(args.seed, i, screen_size, args.reward_shaping)
        for i in range(args.n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecTransposeImage(vec_env)

    model = PPO(
        "CnnPolicy",
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

    callback = None
    if args.checkpoint_freq > 0:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=args.checkpoint_dir,
            name_prefix=save_path.name,
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=args.progress_bar,
    )
    model.save(str(save_path))
    vec_env.close()

    eval_env = DummyVecEnv([make_env(args.seed + 10_000, 0, screen_size, False)])
    eval_env = VecTransposeImage(eval_env)
    eval_lengths = []
    for _ in range(args.eval_episodes):
        obs = eval_env.reset()
        done = [False]
        steps = 0
        while not done[0] and steps < args.eval_max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env.step(action)
            steps += 1
        eval_lengths.append(steps)
    eval_env.close()

    print(f"Training finished on {ENV_ID}. Model saved to: {save_path}.zip")
    print(f"Quick eval episode lengths: {eval_lengths}")


if __name__ == "__main__":
    main()