"""
PPO测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import time

from envs.alpha_rl_env import AlphaRobotEnv
from envs.safety_wrapper import SafetyWrapper
from PPO.ppo_trainer import PPOTrainer
from common.utils import set_random_seed


def test_policy(trainer, env, num_episodes=10, render=True, verbose=True):
    """测试训练好的策略"""
    
    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_count': 0,
        'safety_violations': [],
        'distances': []
    }
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        safety_violations = 0
        min_distance = float('inf')
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Target position: {env.target_position}")
        
        done = False
        while not done and episode_length < env.max_steps:
            # 选择动作
            with torch.no_grad():
                action = trainer.select_action(state, evaluate=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 更新统计
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            # 记录安全违规
            if 'safety_penalty' in info and info['safety_penalty'] > 0:
                safety_violations += 1
                
            # 记录最小距离
            if 'distance' in info:
                min_distance = min(min_distance, info['distance'])
                
            # 显示进度
            if verbose and episode_length % 100 == 0:
                print(f"  Step {episode_length}: "
                      f"Distance = {info.get('distance', 0):.3f}m, "
                      f"Reward = {episode_reward:.2f}")
                      
            state = next_state
            
            if render:
                env.render()
                time.sleep(0.01)  # 控制渲染速度
        
        # 记录结果
        results['episode_rewards'].append(episode_reward)
        results['episode_lengths'].append(episode_length)
        results['distances'].append(min_distance)
        results['safety_violations'].append(safety_violations)
        
        success = info.get('success', False)
        if success:
            results['success_count'] += 1
            
        if verbose:
            print(f"\nEpisode Summary:")
            print(f"  Success: {'Yes' if success else 'No'}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Episode Length: {episode_length}")
            print(f"  Min Distance: {min_distance:.3f}m")
            print(f"  Safety Violations: {safety_violations}")
    
    # 计算统计
    success_rate = results['success_count'] / num_episodes
    avg_reward = np.mean(results['episode_rewards'])
    std_reward = np.std(results['episode_rewards'])
    avg_length = np.mean(results['episode_lengths'])
    avg_distance = np.mean(results['distances'])
    avg_violations = np.mean(results['safety_violations'])
    
    print(f"\n{'='*50}")
    print("Test Results Summary")
    print(f"{'='*50}")
    print(f"Success Rate: {success_rate:.2%} ({results['success_count']}/{num_episodes})")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Length: {avg_length:.1f} steps")
    print(f"Average Min Distance: {avg_distance:.3f}m")
    print(f"Average Safety Violations: {avg_violations:.1f}")
    print(f"{'='*50}")
    
    return results


def visualize_workspace(trainer, env, num_positions=20):
    """可视化工作空间中的到达能力"""
    print("\nVisualizing workspace reachability...")
    
    # 在工作空间中采样位置
    positions = []
    successes = []
    
    for i in range(num_positions):
        # 采样目标位置
        target = env._sample_valid_target()
        positions.append(target)
        
        # 测试是否能到达
        state, _ = env.reset(target_position=target)
        
        for _ in range(200):  # 给定步数尝试到达
            action = trainer.select_action(state, evaluate=True)
            state, _, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
                
        success = info.get('success', False)
        successes.append(success)
        
        print(f"  Position {i+1}/{num_positions}: "
              f"{target} - {'Success' if success else 'Failed'}")
    
    # 统计结果
    reachability_rate = sum(successes) / len(successes)
    print(f"\nWorkspace Reachability: {reachability_rate:.2%}")
    
    return positions, successes


def main():
    parser = argparse.ArgumentParser(description='Test trained PPO policy')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--safety', action='store_true', default=True,
                        help='Enable safety wrapper')
    parser.add_argument('--workspace-test', action='store_true',
                        help='Test workspace reachability')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 创建环境
    print("Creating environment...")
    env = AlphaRobotEnv(
        render_mode="human" if not args.no_render else None,
        dense_reward=True
    )
    
    # 添加安全包装器
    if args.safety:
        env = SafetyWrapper(env)
    
    # 创建训练器并加载模型
    print(f"Loading model from: {args.model}")
    trainer = PPOTrainer(env)
    trainer.load_checkpoint(args.model)
    
    # 测试策略
    results = test_policy(
        trainer, env, 
        num_episodes=args.episodes,
        render=not args.no_render,
        verbose=True
    )
    
    # 工作空间测试
    if args.workspace_test:
        visualize_workspace(trainer, env)
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()