"""
PPO训练脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from datetime import datetime

from envs.alpha_rl_env import AlphaRobotEnv
from envs.safety_wrapper import SafetyWrapper
from PPO.ppo_trainer import PPOTrainer
from common.utils import set_random_seed


def main():
    parser = argparse.ArgumentParser(description='Train Alpha Robot with PPO')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    
    # 环境参数
    parser.add_argument('--dense-reward', action='store_true', default=True,
                        help='Use dense reward')
    parser.add_argument('--curriculum', action='store_true',
                        help='Use curriculum learning')
    parser.add_argument('--safety', action='store_true', default=True,
                        help='Enable safety wrapper')
    
    # PPO参数
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--rollout-length', type=int, default=2048,
                        help='Rollout length')
    parser.add_argument('--ppo-epochs', type=int, default=10,
                        help='PPO epochs')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                        help='PPO clip parameter')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 创建环境
    print("Creating environment...")
    env = AlphaRobotEnv(
        render_mode="human" if args.render else None,
        dense_reward=args.dense_reward,
        curriculum_learning=args.curriculum
    )
    
    # 添加安全包装器
    if args.safety:
        print("Adding safety wrapper...")
        env = SafetyWrapper(env)
    
    # PPO配置
    ppo_config = {
        'lr_actor': args.lr,
        'lr_critic': args.lr,
        'rollout_length': args.rollout_length,
        'ppo_epochs': args.ppo_epochs,
        'clip_epsilon': args.clip_epsilon,
        'eval_freq': 1000,
        'save_freq': 5000,
        'checkpoint_freq': 10000,
        'eval_episodes': 20,
        'render_eval': args.render
    }
    
    # 加载自定义配置
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = yaml.safe_load(f)
            ppo_config.update(custom_config)
    
    # 打印配置
    print("\n" + "="*50)
    print("PPO Training Configuration")
    print("="*50)
    print(f"Environment: AlphaRobotEnv")
    print(f"State Space: {env.observation_space.shape}")
    print(f"Action Space: {env.action_space.shape}")
    print(f"Episodes: {args.episodes}")
    print(f"Random Seed: {args.seed}")
    print(f"Dense Reward: {args.dense_reward}")
    print(f"Curriculum Learning: {args.curriculum}")
    print(f"Safety Wrapper: {args.safety}")
    print("\nPPO Parameters:")
    for key, value in ppo_config.items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")
    
    # 创建训练器
    trainer = PPOTrainer(env, ppo_config)
    
    # 开始训练
    try:
        trainer.train(args.episodes)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # 保存最终模型
        trainer.save_checkpoint('final')
        trainer.plot_training_curves()
        
    # 关闭环境
    env.close()
    
    print("\nTraining completed!")
    print(f"Results saved in: {trainer.base_dir}")


if __name__ == "__main__":
    main()