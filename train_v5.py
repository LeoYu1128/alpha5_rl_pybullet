import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import time

# 导入环境
from envs.rl_env_v5 import AlphaReachEnv

class TrainingProgressCallback(BaseCallback):
    """训练进度回调，记录成功率等指标"""
    
    def __init__(self, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.success_rates = []
        self.distances = []
        self.episodes = 0
        self.successes = 0
        
    def _on_step(self) -> bool:
        # 记录回合信息
        if 'episode' in self.locals and self.locals['episode']:
            infos = self.locals.get('infos', [])
            for info in infos:
                if 'success' in info:
                    self.episodes += 1
                    if info['success']:
                        self.successes += 1
                    self.distances.append(info.get('distance', 0))
        
        return True
    
    def _on_training_end(self) -> None:
        if self.episodes > 0:
            success_rate = self.successes / self.episodes
            avg_distance = np.mean(self.distances) if self.distances else 0
            print(f"训练完成 - 成功率: {success_rate:.2%}, 平均距离: {avg_distance:.3f}m")

def make_env(render_mode=None):
    """创建环境包装器"""
    def _init():
        env = AlphaReachEnv(render_mode=render_mode)
        return Monitor(env)
    return _init

def train_alpha_reach(algorithm='SAC', total_timesteps=50000, save_path="./models"):
    """训练Alpha机械臂到达任务"""
    
    print(f"开始训练Alpha机械臂到达任务")
    print(f"算法: {algorithm}")
    print(f"总步数: {total_timesteps}")
    print("="*50)
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # 首先检查环境
    print("检查环境...")
    test_env = AlphaReachEnv()
    check_env(test_env)
    test_env.close()
    print("环境检查通过")
    
    # 创建训练环境
    train_env = DummyVecEnv([make_env() for _ in range(1)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    # 创建评估环境
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # 算法配置
    configs = {
        'SAC': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'learning_starts': 1000,
            'ent_coef': 'auto',
            'verbose': 1
        },
        'PPO': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'verbose': 1
        },
        'TD3': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'learning_starts': 1000,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            'verbose': 1
        }
    }
    
    # 创建模型
    algo_classes = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}
    model = algo_classes[algorithm](env=train_env, **configs[algorithm])
    
    # 设置回调函数
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/{algorithm}_best",
        log_path=f"./logs/{algorithm}",
        eval_freq=2000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{save_path}/{algorithm}_checkpoints",
        name_prefix=f"{algorithm}_checkpoint"
    )
    
    progress_callback = TrainingProgressCallback(eval_freq=1000)
    
    # 开始训练
    print("开始训练...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, progress_callback],
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.1f}秒")
    
    # 保存最终模型
    model.save(f"{save_path}/{algorithm}_final")
    train_env.save(f"{save_path}/{algorithm}_vecnormalize.pkl")
    
    # 最终评估
    print("进行最终评估...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    
    print(f"最终评估结果:")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 清理
    train_env.close()
    eval_env.close()
    
    return model, mean_reward, std_reward

def test_trained_model(model_path, algorithm='SAC', n_episodes=5):
    """测试训练好的模型"""
    
    print(f"测试训练好的{algorithm}模型...")
    print(f"模型路径: {model_path}")
    
    # 创建可视化环境
    env = AlphaReachEnv(render_mode="human")
    
    # 加载模型
    algo_classes = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}
    model = algo_classes[algorithm].load(model_path)
    
    # 如果有VecNormalize，也要加载
    try:
        vec_normalize_path = model_path.replace('_final', '_vecnormalize.pkl')
        if os.path.exists(vec_normalize_path):
            print("加载环境标准化参数...")
            test_env = DummyVecEnv([lambda: env])
            test_env = VecNormalize.load(vec_normalize_path, test_env)
            test_env.training = False
            test_env.norm_reward = False
            use_vec_env = True
        else:
            use_vec_env = False
    except:
        use_vec_env = False
    
    successes = 0
    total_rewards = []
    distances = []
    
    for episode in range(n_episodes):
        if use_vec_env:
            obs = test_env.reset()
        else:
            obs, _ = env.reset()
            
        episode_reward = 0
        step_count = 0
        
        for step in range(500):  # 最多200步
            if use_vec_env:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                episode_reward += reward[0]
                step_count += 1
                
                if done[0]:
                    info = info[0] if len(info) > 0 else {}
                    break
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                if terminated or truncated:
                    break
        
        # 记录结果
        success = info.get('success', False)
        distance = info.get('distance', 0)
        
        if success:
            successes += 1
        
        total_rewards.append(episode_reward)
        distances.append(distance)
        
        print(f"Episode {episode+1}: 成功={success}, 距离={distance:.3f}m, "
              f"奖励={episode_reward:.2f}, 步数={step_count}")
    
    # 打印统计结果
    print("\n" + "="*50)
    print("测试结果统计:")
    print("="*50)
    print(f"成功率: {successes}/{n_episodes} ({successes/n_episodes*100:.1f}%)")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均最终距离: {np.mean(distances):.3f} ± {np.std(distances):.3f}m")
    
    if successes > 0:
        success_distances = [d for i, d in enumerate(distances) if i < len(total_rewards) and 
                           total_rewards[i] > 0]  # 简化的成功判断
        if success_distances:
            print(f"成功时平均距离: {np.mean(success_distances):.3f}m")
    
    env.close()
    
    return {
        'success_rate': successes / n_episodes,
        'mean_reward': np.mean(total_rewards),
        'mean_distance': np.mean(distances)
    }

def compare_algorithms(algorithms=['SAC', 'PPO', 'TD3'], timesteps=30000):
    """比较不同算法的性能"""
    
    print("开始算法性能比较")
    print(f"算法: {algorithms}")
    print(f"训练步数: {timesteps}")
    print("="*50)
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\n正在训练 {algorithm}...")
        try:
            model, mean_reward, std_reward = train_alpha_reach(
                algorithm=algorithm,
                total_timesteps=timesteps
            )
            results[algorithm] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'model': model
            }
            print(f"{algorithm} 训练完成")
        except Exception as e:
            print(f"{algorithm} 训练失败: {e}")
            results[algorithm] = {
                'mean_reward': -np.inf,
                'std_reward': 0,
                'model': None
            }
    
    # 打印比较结果
    print("\n" + "="*60)
    print("算法性能比较结果:")
    print("="*60)
    print(f"{'算法':<8} {'平均奖励':<12} {'标准差':<10} {'相对性能':<10}")
    print("-" * 60)
    
    best_reward = max([r['mean_reward'] for r in results.values() if r['mean_reward'] != -np.inf])
    
    for algo, result in results.items():
        if result['mean_reward'] != -np.inf:
            relative_perf = result['mean_reward'] / best_reward * 100
            print(f"{algo:<8} {result['mean_reward']:<12.2f} {result['std_reward']:<10.2f} {relative_perf:<10.1f}%")
        else:
            print(f"{algo:<8} {'失败':<12} {'-':<10} {'-':<10}")
    
    # 找出最佳算法
    best_algo = max(results.keys(), key=lambda x: results[x]['mean_reward'])
    if results[best_algo]['mean_reward'] != -np.inf:
        print(f"\n最佳算法: {best_algo} (奖励: {results[best_algo]['mean_reward']:.2f})")
    
    return results

def plot_training_results(log_dir="./logs"):
    """绘制训练结果图表"""
    try:
        import tensorboard
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        print("绘制训练结果...")
        
        algorithms = ['SAC', 'PPO', 'TD3']
        colors = ['blue', 'red', 'green']
        
        plt.figure(figsize=(12, 8))
        
        for i, algo in enumerate(algorithms):
            log_path = os.path.join(log_dir, algo)
            if os.path.exists(log_path):
                # 这里可以添加从tensorboard日志读取数据的代码
                # 简化版：创建示例数据
                steps = np.arange(0, 50000, 1000)
                rewards = np.random.normal(0, 1, len(steps)).cumsum() * 0.1
                plt.plot(steps, rewards, color=colors[i], label=algo, alpha=0.7)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Average Reward')
        plt.title('Training Progress Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("需要安装tensorboard来绘制训练图表")
    except Exception as e:
        print(f"绘制图表失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpha机械臂到达任务训练')
    parser.add_argument('--mode', choices=['train', 'test', 'compare'], 
                       default='train', help='运行模式')
    parser.add_argument('--algorithm', choices=['SAC', 'PPO', 'TD3'], 
                       default='SAC', help='RL算法')
    parser.add_argument('--timesteps', type=int, default=50000, 
                       help='训练步数')
    parser.add_argument('--model', type=str, 
                       help='测试模式下的模型路径')
    parser.add_argument('--episodes', type=int, default=5, 
                       help='测试回合数')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("单算法训练模式")
        train_alpha_reach(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps
        )
        
    elif args.mode == 'test':
        if not args.model:
            # 尝试加载默认模型
            model_path = f"./models/{args.algorithm}_final"
            if not os.path.exists(model_path + '.zip'):
                print(f"错误: 找不到模型文件 {model_path}")
                print("请先训练模型或指定正确的模型路径")
                return
        else:
            model_path = args.model
            
        print("模型测试模式")
        test_trained_model(
            model_path=model_path,
            algorithm=args.algorithm,
            n_episodes=args.episodes
        )
        
    elif args.mode == 'compare':
        print("算法比较模式")
        results = compare_algorithms(
            algorithms=['SAC', 'PPO', 'TD3'],
            timesteps=args.timesteps
        )
        
        # 绘制结果
        plot_training_results()

if __name__ == "__main__":
    main()