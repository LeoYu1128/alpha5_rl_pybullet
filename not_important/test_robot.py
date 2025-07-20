#!/usr/bin/env python3
"""
Alpha Robot 训练结果测试脚本
专门用来观察训练好的机器人行为
"""

import os
import sys
import numpy as np
import torch
import time
import pybullet as p
from envs.alpha_rl_env import AlphaRobotEnv, ActorCritic

def find_best_model():
    """查找最佳模型文件"""
    model_files = []
    
    # 查找所有可能的模型文件
    possible_files = [
        'best_alpha_robot_model.pth',
        'alpha_robot_model_90.pth',
        'alpha_robot_model_80.pth',
        'alpha_robot_model_70.pth',
        'alpha_robot_model_60.pth',
        'alpha_robot_model_50.pth',
    ]
    
    for file in possible_files:
        if os.path.exists(file):
            model_files.append(file)
    
    # 如果没有找到预定义的文件，搜索所有.pth文件
    if not model_files:
        for file in os.listdir('.'):
            if file.endswith('.pth') and 'alpha_robot' in file:
                model_files.append(file)
    
    if not model_files:
        print("❌ 没有找到任何模型文件")
        print("请确保以下文件存在：")
        print("   - best_alpha_robot_model.pth")
        print("   - alpha_robot_model_XX.pth")
        return None
    
    # 优先选择最佳模型
    if 'best_alpha_robot_model.pth' in model_files:
        return 'best_alpha_robot_model.pth'
    else:
        return model_files[0]

def test_trained_robot(model_path=None, num_episodes=5, slow_motion=True):
    """测试训练好的机器人"""
    
    print("🤖 Alpha Robot 行为测试")
    print("=" * 40)
    
    # 查找模型文件
    if model_path is None:
        model_path = find_best_model()
        if model_path is None:
            return
    
    print(f"📂 加载模型: {model_path}")
    
    # 创建环境（GUI模式，可以看到机器人）
    print("🏗️  创建测试环境...")
    env = AlphaRobotEnv(render_mode='human')
    
    # 获取环境信息
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    print(f"📊 环境信息:")
    print(f"   状态空间大小: {state_size}")
    print(f"   动作空间大小: {action_size}")
    print(f"   可移动关节数: {len(env.movable_joints)}")
    
    # 创建策略网络
    policy = ActorCritic(state_size, action_size)
    
    # 加载训练好的模型
    try:
        policy.load_state_dict(torch.load(model_path, map_location='cpu'))
        policy.eval()
        print("✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        env.close()
        return
    
    print(f"\n🎮 开始测试 {num_episodes} 个回合...")
    print("🎯 观察机器人如何移动去触碰红色目标球")
    if slow_motion:
        print("🐌 慢速模式：便于观察机器人行为")
    print("按 Ctrl+C 可以随时停止测试\n")
    
    total_rewards = []
    success_count = 0
    
    try:
        for episode in range(num_episodes):
            print(f"📍 回合 {episode + 1}/{num_episodes}")
            
            # 重置环境
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
            max_steps = 1000
            
            print(f"   🎯 目标位置: ({env.target_position[0]:.2f}, {env.target_position[1]:.2f}, {env.target_position[2]:.2f})")
            
            while steps < max_steps:
                # 获取动作
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                with torch.no_grad():
                    action_mean, action_std, state_value = policy(state_tensor)
                
                # 使用确定性动作（均值）来观察学到的行为
                action = action_mean.cpu().numpy()[0]
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                
                # 每50步显示一次信息
                if steps % 50 == 0:
                    # 获取机器人当前位置
                    num_links = p.getNumJoints(env.robot_id)
                    if num_links > 0:
                        try:
                            end_effector_state = p.getLinkState(env.robot_id, num_links - 1)
                            end_pos = end_effector_state[0]
                        except:
                            end_pos = p.getBasePositionAndOrientation(env.robot_id)[0]
                    else:
                        end_pos = p.getBasePositionAndOrientation(env.robot_id)[0]
                    
                    distance = np.linalg.norm(np.array(end_pos) - env.target_position)
                    print(f"   步数: {steps:3d}, 距离: {distance:.3f}, 奖励: {reward:.2f}")
                
                # 慢速模式
                if slow_motion:
                    time.sleep(0.05)  # 50ms延迟，便于观察
                
                if done:
                    if terminated:
                        print(f"   🎉 成功！机器人达到了目标位置")
                        success_count += 1
                    else:
                        print(f"   ⏰ 回合结束（达到最大步数）")
                    break
                
                state = next_state
            
            total_rewards.append(episode_reward)
            print(f"   回合奖励: {episode_reward:.2f}")
            print(f"   步数: {steps}")
            print()
            
            # 回合间暂停
            if episode < num_episodes - 1:
                print("   等待3秒后开始下一回合...")
                time.sleep(3)
    
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    
    # 统计结果
    print("📊 测试结果统计:")
    print(f"   总回合数: {len(total_rewards)}")
    print(f"   成功次数: {success_count}")
    print(f"   成功率: {success_count/len(total_rewards)*100:.1f}%")
    print(f"   平均奖励: {np.mean(total_rewards):.2f}")
    print(f"   最佳奖励: {np.max(total_rewards):.2f}")
    print(f"   最差奖励: {np.min(total_rewards):.2f}")
    
    env.close()

def interactive_test():
    """交互式测试模式"""
    print("🎮 交互式测试模式")
    print("=" * 40)
    
    # 查找模型
    model_path = find_best_model()
    if model_path is None:
        return
    
    while True:
        print("\n请选择测试选项：")
        print("1. 慢速观察 (推荐)")
        print("2. 正常速度")
        print("3. 单回合详细测试")
        print("4. 连续测试多回合")
        print("5. 退出")
        
        choice = input("输入选项 (1-5): ").strip()
        
        if choice == '1':
            test_trained_robot(model_path, num_episodes=3, slow_motion=True)
        elif choice == '2':
            test_trained_robot(model_path, num_episodes=3, slow_motion=False)
        elif choice == '3':
            test_trained_robot(model_path, num_episodes=1, slow_motion=True)
        elif choice == '4':
            num = input("输入回合数 (默认5): ").strip()
            try:
                num_episodes = int(num) if num else 5
            except ValueError:
                num_episodes = 5
            test_trained_robot(model_path, num_episodes=num_episodes, slow_motion=False)
        elif choice == '5':
            print("👋 再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

def analyze_model_behavior():
    """分析模型行为"""
    print("🔍 模型行为分析")
    print("=" * 40)
    
    model_path = find_best_model()
    if model_path is None:
        return
    
    # 创建环境
    env = AlphaRobotEnv(render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    # 加载模型
    policy = ActorCritic(state_size, action_size)
    policy.load_state_dict(torch.load(model_path, map_location='cpu'))
    policy.eval()
    
    print("🧠 分析模型对不同状态的反应...")
    
    # 测试不同的目标位置
    test_targets = [
        [0.1, 0.0, 0.1],   # 近距离
        [0.2, 0.0, 0.2],   # 中距离
        [0.2, 0.1, 0.1],   # 侧边
        [0.2, -0.1, 0.2],  # 另一侧
    ]
    
    for i, target in enumerate(test_targets):
        print(f"\n📍 测试目标位置 {i+1}: {target}")
        
        # 手动设置目标位置
        env.target_position = np.array(target)
        p.resetBasePositionAndOrientation(env.target_id, target, [0, 0, 0, 1])
        1
        state, _ = env.reset()
        
        # 观察前几步的动作
        for step in range(10):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_mean, action_std, state_value = policy(state_tensor)
            
            action = action_mean.cpu().numpy()[0]
            print(f"   步数 {step+1}: 动作 = {action[:3]}, 价值 = {state_value.item():.3f}")
            
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            
            time.sleep(0.1)
    
    env.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'analyze':
            analyze_model_behavior()
        elif sys.argv[1] == 'quick':
            test_trained_robot(num_episodes=1, slow_motion=True)
        else:
            print("用法:")
            print("  python test_robot.py          # 交互式测试")
            print("  python test_robot.py quick    # 快速测试一回合")
            print("  python test_robot.py analyze  # 分析模型行为")
    else:
        interactive_test()