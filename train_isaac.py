"""
Isaac Lab Training Script for Alpha Underwater Reach
GPU加速训练 - 针对RTX 3070优化
"""

import argparse
import os
from datetime import datetime

import torch
import gymnasium as gym

# Isaac Lab导入
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.utils import parse_env_cfg

# RL库导入 (Stable-Baselines3)
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

# 导入环境配置
from alpha_reach_env_cfg import AlphaReachEnvCfg_RTX3070


def create_isaac_env(cfg: ManagerBasedRLEnvCfg, num_envs: int = 2048):
    """
    创建Isaac Lab向量化环境
    
    Args:
        cfg: 环境配置
        num_envs: 并行环境数 (RTX 3070: 2048推荐)
    
    Returns:
        env: 向量化环境
    """
    print(f"\n{'='*60}")
    print("创建Isaac Lab环境")
    print(f"{'='*60}")
    print(f"并行环境数: {num_envs}")
    print(f"GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")
    
    # 更新配置
    cfg.scene.num_envs = num_envs
    
    # 创建环境
    env = gym.make("Isaac-Alpha-Reach-v0", cfg=cfg)
    
    # 包装为Stable-Baselines3兼容格式
    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
    env = Sb3VecEnvWrapper(env)
    
    print(f"✓ 环境创建成功")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  并行数: {env.num_envs}")
    
    return env


def train_sac(
    env,
    total_timesteps: int = 1_000_000,
    save_dir: str = "experiments_isaac",
    algorithm: str = "SAC",
):
    """
    使用SAC训练
    
    Args:
        env: Isaac Lab环境
        total_timesteps: 总训练步数
        save_dir: 保存目录
        algorithm: 算法名称
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(save_dir, f"{algorithm}_isaac_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"开始训练: {algorithm}")
    print(f"{'='*60}")
    print(f"总步数: {total_timesteps:,}")
    print(f"实验目录: {exp_dir}")
    print(f"{'='*60}\n")
    
    # SAC配置 (针对Isaac Lab优化)
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=2048,  # 大batch利用GPU
        tau=0.005,
        gamma=0.98,
        train_freq=4,  # 每4步训练一次
        gradient_steps=4,
        learning_starts=10000,
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
            activation_fn=torch.nn.ReLU,
        ),
        tensorboard_log=os.path.join(exp_dir, "tensorboard"),
        device="cuda",
        verbose=1,
    )
    
    # 配置日志
    logger = configure(exp_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    
    # 回调
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(exp_dir, "checkpoints"),
        name_prefix=algorithm,
    )
    
    # 训练
    print("开始训练...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        progress_bar=True,
    )
    
    # 保存最终模型
    final_model_path = os.path.join(exp_dir, f"{algorithm}_final")
    model.save(final_model_path)
    print(f"\n✓ 最终模型已保存: {final_model_path}")
    
    return model, exp_dir


def train_ppo(
    env,
    total_timesteps: int = 1_000_000,
    save_dir: str = "experiments_isaac",
):
    """
    使用PPO训练 (在Isaac Lab中PPO通常更稳定)
    
    Args:
        env: Isaac Lab环境
        total_timesteps: 总训练步数
        save_dir: 保存目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(save_dir, f"PPO_isaac_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("开始训练: PPO")
    print(f"{'='*60}")
    print(f"总步数: {total_timesteps:,}")
    print(f"实验目录: {exp_dir}")
    print(f"{'='*60}\n")
    
    # PPO配置
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,  # 每次收集2048步
        batch_size=512,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
            activation_fn=torch.nn.ReLU,
        ),
        tensorboard_log=os.path.join(exp_dir, "tensorboard"),
        device="cuda",
        verbose=1,
    )
    
    # 配置日志
    logger = configure(exp_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    
    # 回调
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(exp_dir, "checkpoints"),
        name_prefix="PPO",
    )
    
    # 训练
    print("开始训练...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        progress_bar=True,
    )
    
    # 保存最终模型
    final_model_path = os.path.join(exp_dir, "PPO_final")
    model.save(final_model_path)
    print(f"\n✓ 最终模型已保存: {final_model_path}")
    
    return model, exp_dir


def benchmark_throughput(env, duration_seconds: int = 60):
    """
    性能基准测试
    
    Args:
        env: Isaac Lab环境
        duration_seconds: 测试时长(秒)
    """
    print(f"\n{'='*60}")
    print("GPU性能基准测试")
    print(f"{'='*60}")
    print(f"测试时长: {duration_seconds}秒")
    print(f"并行环境: {env.num_envs}")
    print(f"{'='*60}\n")
    
    import time
    
    obs = env.reset()
    total_steps = 0
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        actions = env.action_space.sample()
        obs, rewards, dones, infos = env.step(actions)
        total_steps += 1
    
    elapsed = time.time() - start_time
    fps = (total_steps * env.num_envs) / elapsed
    
    print(f"\n{'='*60}")
    print("基准测试结果")
    print(f"{'='*60}")
    print(f"总步数: {total_steps * env.num_envs:,}")
    print(f"FPS: {fps:,.0f} (环境步数/秒)")
    print(f"吞吐量: {fps / env.num_envs:.1f}x 实时")
    print(f"{'='*60}\n")
    
    # 与PyBullet对比
    pybullet_fps = 60  # PyBullet大约60 FPS
    speedup = fps / pybullet_fps
    print(f"相比PyBullet加速: {speedup:.0f}x")
    print(f"预计训练时间节省: {(1 - 1/speedup) * 100:.1f}%")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Isaac Lab训练脚本")
    parser.add_argument("--algorithm", choices=["SAC", "PPO"], default="SAC", help="RL算法")
    parser.add_argument("--num_envs", type=int, default=2048, help="并行环境数")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="总训练步数")
    parser.add_argument("--benchmark", action="store_true", help="运行性能测试")
    parser.add_argument("--save_dir", type=str, default="experiments_isaac", help="保存目录")
    
    args = parser.parse_args()
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA,Isaac Lab需要GPU")
        return
    
    print(f"\n{'='*60}")
    print("Isaac Lab训练系统")
    print(f"{'='*60}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"{'='*60}\n")
    
    # 创建环境配置
    cfg = AlphaReachEnvCfg_RTX3070()
    
    # 创建环境
    env = create_isaac_env(cfg, num_envs=args.num_envs)
    
    # 性能测试
    if args.benchmark:
        benchmark_throughput(env, duration_seconds=60)
        env.close()
        return
    
    # 训练
    if args.algorithm == "SAC":
        model, exp_dir = train_sac(env, total_timesteps=args.timesteps, save_dir=args.save_dir)
    else:
        model, exp_dir = train_ppo(env, total_timesteps=args.timesteps, save_dir=args.save_dir)
    
    # 关闭环境
    env.close()
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    print(f"实验目录: {exp_dir}")
    print(f"模型: {exp_dir}/{args.algorithm}_final.zip")
    print(f"TensorBoard: tensorboard --logdir={exp_dir}/tensorboard")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()