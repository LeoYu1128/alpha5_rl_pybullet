"""
Alpha Robot 主训练脚本
支持单独训练和算法比较
"""

import argparse
from envs.alpha_rl_env import AlphaRobotEnv
from alpha_robot_trainer import PPOTrainer, SACTrainer, AlgorithmComparison
import numpy as np
import pybullet as p


def train_single_algorithm(algorithm, num_episodes=1e5, render=False):
    """训练单个算法"""
    # 创建环境
    env = AlphaRobotEnv(render_mode="human" if render else None)
    
    # 算法配置
    configs = {
        'PPO': {
            'lr': 3e-4,
            'ppo_epochs': 10,
            'clip_epsilon': 0.2,
            'rollout_length': 2048,
        },
        'SAC': {
            'lr': 3e-4,
            'batch_size': 256,
            'automatic_entropy_tuning': True,
        }
    }
    
    # 创建训练器
    if algorithm == 'PPO':
        trainer = PPOTrainer(env, configs['PPO'])
    elif algorithm == 'SAC':
        trainer = SACTrainer(env, configs['SAC'])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
        
    print(f"\n{'='*50}")
    print(f"Training {algorithm} on Alpha Robot")
    print(f"{'='*50}")
    print(f"Environment: {env.__class__.__name__}")
    print(f"State Space: {env.observation_space.shape}")
    print(f"Action Space: {env.action_space.shape}")
    print(f"Max Reach: {env.MAX_REACH}m")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*50}\n")
    
    # 训练
    trainer.train(num_episodes)
    
    # 最终评估
    print("\n=== Final Evaluation ===")
    avg_reward, success_rate = trainer.evaluate(num_episodes=20)
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    
    # 保存最终模型
    trainer.save_model(f"alpha_robot_{algorithm}_final.pth")
    trainer.plot_results()
    
    env.close()
    
    return trainer


def compare_algorithms(num_episodes=500, num_seeds=3):
    """比较不同算法"""
    # 环境创建函数
    def env_creator():
        return AlphaRobotEnv(render_mode=None)
    
    # 创建比较器
    comparison = AlgorithmComparison(env_creator)
    
    # 添加算法
    comparison.add_algorithm('PPO', PPOTrainer, {
        'lr': 3e-4,
        'ppo_epochs': 10,
        'eval_freq': 20,
    })
    
    comparison.add_algorithm('SAC', SACTrainer, {
        'lr': 3e-4,
        'batch_size': 256,
        'eval_freq': 20,
    })
    
    # 可以添加更多算法，如TD3
    # comparison.add_algorithm('TD3', TD3Trainer, {...})
    
    print(f"\n{'='*50}")
    print("Comparing RL Algorithms on Alpha Robot")
    print(f"{'='*50}")
    print(f"Episodes per seed: {num_episodes}")
    print(f"Number of seeds: {num_seeds}")
    print(f"Algorithms: {list(comparison.results.keys())}")
    print(f"{'='*50}\n")
    
    # 运行比较
    comparison.run_comparison(num_episodes, num_seeds)
    
    # 绘制结果
    comparison.plot_comparison()


def test_trained_model(model_path, algorithm='PPO', num_episodes=10):
    """测试已训练的模型"""
    # 创建环境
    env = AlphaRobotEnv(render_mode="human")
    
    # 创建训练器并加载模型
    if algorithm == 'PPO':
        trainer = PPOTrainer(env)
    elif algorithm == 'SAC':
        trainer = SACTrainer(env)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
        
    trainer.load_model(model_path)
    
    print(f"\n{'='*50}")
    print(f"Testing {algorithm} Model")
    print(f"Model Path: {model_path}")
    print(f"{'='*50}\n")
    
    # 测试多个episode
    total_reward = 0
    successes = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done and steps < 1000:
            # 选择动作
            action = trainer.select_action(state, evaluate=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            state = next_state
            steps += 1
            
            # 显示进度
            if steps % 100 == 0:
                distance = info.get('distance', 0)
                print(f"  Step {steps}: Distance to target: {distance:.3f}m")
                
        total_reward += episode_reward
        if info.get('success', False):
            successes += 1
            print(f"  ✓ Success! Reward: {episode_reward:.2f}")
        else:
            print(f"  ✗ Failed. Reward: {episode_reward:.2f}")
            
    # 统计结果
    print(f"\n{'='*50}")
    print(f"Test Results:")
    print(f"  Average Reward: {total_reward/num_episodes:.2f}")
    print(f"  Success Rate: {successes/num_episodes:.2%}")
    print(f"{'='*50}")
    
    env.close()


def interactive_control():
    """交互式控制（用于调试）"""
    import pygame
    
    # 创建环境
    env = AlphaRobotEnv(render_mode="human")
    
    print("\n=== Interactive Control Mode ===")
    print("Use arrow keys to control joints:")
    print("  1-5: Select joint")
    print("  ↑/↓: Increase/Decrease joint angle")
    print("  R: Reset environment")
    print("  G: Go to random target")
    print("  ESC: Exit")
    print("================================\n")
    
    pygame.init()
    clock = pygame.time.Clock()
    
    state, _ = env.reset()
    selected_joint = 0
    action = np.zeros(5)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state, _ = env.reset()
                    action = np.zeros(5)
                elif event.key == pygame.K_g:
                    env.target_position = env._sample_valid_target()
                    p.resetBasePositionAndOrientation(
                        env.target_id,
                        env.target_position,
                        [0, 0, 0, 1]
                    )
                elif pygame.K_1 <= event.key <= pygame.K_5:
                    selected_joint = event.key - pygame.K_1
                    print(f"Selected joint {selected_joint + 1}")
                    
        # 获取按键状态
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[selected_joint] = min(1.0, action[selected_joint] + 0.05)
        elif keys[pygame.K_DOWN]:
            action[selected_joint] = max(-1.0, action[selected_joint] - 0.05)
            
        # 执行动作
        state, reward, terminated, truncated, info = env.step(action)
        
        # 显示信息
        distance = info.get('distance', 0)
        print(f"\rDistance: {distance:.3f}m, Reward: {reward:.2f}", end='')
        
        if terminated:
            print("\n✓ Target reached!")
            state, _ = env.reset()
            action = np.zeros(5)
            
        clock.tick(30)  # 30 FPS
        
    pygame.quit()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Alpha Robot RL Training')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'compare', 'test', 'interactive'],
                        help='Execution mode')
    parser.add_argument('--algorithm', type=str, default='PPO',
                        choices=['PPO', 'SAC'],
                        help='RL algorithm to use')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render during training')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to saved model (for test mode)')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of seeds for comparison')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_single_algorithm(args.algorithm, args.episodes, args.render)
    elif args.mode == 'compare':
        compare_algorithms(args.episodes, args.seeds)
    elif args.mode == 'test':
        if args.model is None:
            print("Error: --model path required for test mode")
        else:
            test_trained_model(args.model, args.algorithm)
    elif args.mode == 'interactive':
        interactive_control()