#!/usr/bin/env python3
"""
Alpha Robot 强化学习训练启动脚本
简化版本，适合初学者使用
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'pybullet', 'gymnasium', 'torch', 'numpy', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def simple_train():
    """简化的训练函数"""
    print("=" * 50)
    print("🤖 Alpha Robot 强化学习训练")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 导入训练模块（放在这里避免依赖问题）
    try:
        from envs.alpha_rl_env import AlphaRobotEnv, PPOTrainer
        print("✅ 成功导入训练模块")
    except ImportError as e:
        print(f"❌ 导入训练模块失败: {e}")
        print("请确保 alpha_rl_env.py 文件存在于当前目录")
        return
    
    # 训练配置
    config = {
        'render_mode': 'human',  # 可视化训练过程
        'num_iterations': 100,   # 训练迭代次数
        'steps_per_iteration': 512,  # 每次迭代的步数
        'save_interval': 10,     # 保存模型的间隔
        'test_interval': 5,      # 测试策略的间隔
    }
    
    print("📋 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    try:
        # 创建环境
        print("🏗️  创建训练环境...")
        env = AlphaRobotEnv(render_mode=config['render_mode'])
        
        # 获取环境信息
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        
        print(f"📊 环境信息:")
        print(f"   状态空间大小: {state_size}")
        print(f"   动作空间大小: {action_size}")
        print(f"   可移动关节数: {len(env.movable_joints)}")
        print()
        
        # 创建训练器
        print("🧠 创建PPO训练器...")
        trainer = PPOTrainer(env, state_size, action_size)
        
        # 训练历史
        rewards_history = []
        best_reward = -float('inf')
        
        print("🚀 开始训练...")
        print("提示: 你可以看到机器人在GUI中学习如何达到红色目标球")
        print("按 Ctrl+C 可以随时停止训练")
        print()
        
        for iteration in range(config['num_iterations']):
            print(f"📈 迭代 {iteration + 1}/{config['num_iterations']}")
            
            # 收集经验
            print("   收集经验中...")
            trainer.collect_experience(config['steps_per_iteration'])
            
            # 训练
            print("   更新策略中...")
            trainer.train()
            
            # 测试策略
            if iteration % config['test_interval'] == 0:
                print("   测试当前策略...")
                test_reward = test_policy(env, trainer.policy)
                rewards_history.append(test_reward)
                
                # 保存最佳模型
                if test_reward > best_reward:
                    best_reward = test_reward
                    torch.save(trainer.policy.state_dict(), 'best_alpha_robot_model.pth')
                    print(f"   🎉 新的最佳模型! 奖励: {test_reward:.2f}")
                else:
                    print(f"   当前奖励: {test_reward:.2f} (最佳: {best_reward:.2f})")
            
            # 保存检查点
            if iteration % config['save_interval'] == 0:
                torch.save(trainer.policy.state_dict(), f'alpha_robot_checkpoint_{iteration}.pth')
                print(f"   💾 已保存检查点: alpha_robot_checkpoint_{iteration}.pth")
            
            print()
        
        print("🎊 训练完成!")
        print(f"最佳奖励: {best_reward:.2f}")
        
        # 绘制训练曲线
        if rewards_history:
            plt.figure(figsize=(12, 6))
            plt.plot(rewards_history, 'b-', linewidth=2)
            plt.title('Alpha Robot Training Progress', fontsize=16)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Test Reward', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('alpha_robot_training_progress.png', dpi=300)
            print("📊 训练曲线已保存为: alpha_robot_training_progress.png")
            plt.show()
        
        # 最终测试
        print("\n🎯 最终测试...")
        final_reward = test_policy(env, trainer.policy, num_episodes=10)
        print(f"最终平均奖励: {final_reward:.2f}")
        
        env.close()
        
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        if 'env' in locals():
            env.close()
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        if 'env' in locals():
            env.close()

def test_policy(env, policy, num_episodes=3):
    """测试策略性能"""
    total_reward = 0
    
    for episode in range(num_episodes):
        try:
            state, _ = env.reset()
        except:
            state = np.zeros(env.observation_space.shape[0])
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 1000:  # 限制最大步数
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_mean, _, _ = policy(state_tensor)
            
            action = action_mean.cpu().numpy()[0]
            
            try:
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
            except:
                done = True
                break
        
        total_reward += episode_reward
    
    return total_reward / num_episodes

def load_and_test_model():
    """加载已训练的模型并测试"""
    print("🔄 加载已训练的模型进行测试...")
    
    # 导入所需模块
    try:
        from envs.alpha_rl_env import AlphaRobotEnv, ActorCritic
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        return
    
    # 查找可用的模型文件
    model_files = []
    for file in os.listdir('.'):
        if file.endswith('.pth') and 'alpha_robot' in file:
            model_files.append(file)
    
    if not model_files:
        print("❌ 没有找到已训练的模型文件")
        return
    
    # 选择模型
    if 'best_alpha_robot_model.pth' in model_files:
        model_path = 'best_alpha_robot_model.pth'
    else:
        model_path = model_files[0]
    
    print(f"📂 加载模型: {model_path}")
    
    try:
        # 创建环境和策略
        env = AlphaRobotEnv(render_mode='human')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        
        policy = ActorCritic(state_size, action_size)
        policy.load_state_dict(torch.load(model_path))
        policy.eval()
        
        # 测试模型
        print("🎮 测试模型性能...")
        test_reward = test_policy(env, policy, num_episodes=5)
        print(f"平均奖励: {test_reward:.2f}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def show_help():
    """显示帮助信息"""
    print("🤖 Alpha Robot 强化学习训练脚本")
    print()
    print("用法:")
    print("  python simple_train_script.py        # 开始训练")
    print("  python simple_train_script.py test   # 测试已训练的模型")
    print("  python simple_train_script.py help   # 显示帮助信息")
    print()
    print("文件说明:")
    print("  - alpha_rl_env.py: 环境和算法实现")
    print("  - simple_train_script.py: 训练启动脚本（推荐使用）")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            load_and_test_model()
        elif sys.argv[1] == 'help':
            show_help()
        else:
            print("❌ 未知参数，使用 'help' 查看帮助")
    else:
        simple_train()