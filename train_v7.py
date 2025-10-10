import os
import numpy as np
import json
import shutil
from datetime import datetime
import glob

# 设置环境变量避免OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 避免Matplotlib在受限环境下写入HOME目录失败
default_mpl_dir = os.path.join(os.path.dirname(__file__), '.mplconfig')
os.environ.setdefault('MPLCONFIGDIR', default_mpl_dir)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib.pyplot as plt
import pybullet as p
from PIL import Image
from pathlib import Path

# ✅ 导入SB3和SB3-Contrib
from stable_baselines3 import SAC
from sb3_contrib import TQC, CrossQ

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import time

# ✅ 导入ComprehensiveMonitor
from monitor_callbacks import ComprehensiveMonitor

# 导入环境
from envs.rl_env_v9 import AlphaReachEnv


def create_experiment_folder(algorithm, timesteps):
    """创建实验文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{algorithm}_{timesteps}steps_{timestamp}"
    exp_dir = os.path.join("experiments", exp_name)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)

    return exp_dir, exp_name


def save_experiment_config(exp_dir, algorithm, config, timesteps):
    """保存实验配置"""
    exp_config = {
        "algorithm": algorithm,
        "timesteps": timesteps,
        "timestamp": datetime.now().isoformat(),
        "config": config
    }
    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(exp_config, f, indent=2)


class TrainingProgressCallback(BaseCallback):
    """训练进度回调,记录成功率等指标"""
    def __init__(self, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.success_rates = []
        self.distances = []
        self.episodes = 0
        self.successes = 0
        
    def _on_step(self) -> bool:
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


def make_env(idx: int, log_dir: str, render_mode=None):
    """创建环境包装器"""
    def _init():
        env = AlphaReachEnv(render_mode=render_mode)
        filename = os.path.join(log_dir, f"monitor_{idx}.monitor.csv")
        return Monitor(env, filename=filename, info_keywords=("success", "distance"))
    return _init


# ===== ✅ 核心修改:训练函数 =====
def train_alpha_reach(
    algorithm='SAC',
    total_timesteps=300000,
    num_envs=4,
    auto_visualize=True
):
    """训练Alpha机械臂到达任务 - 支持SAC/TQC/CrossQ"""

    print(f"开始训练Alpha机械臂到达任务")
    print(f"算法: {algorithm}")
    print(f"总步数: {total_timesteps}")
    print("="*50)

    # 创建实验文件夹
    exp_dir, exp_name = create_experiment_folder(algorithm, total_timesteps)
    print(f"实验文件夹: {exp_dir}")

    save_path = os.path.join(exp_dir, "models")
    log_path = os.path.join(exp_dir, "logs")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # 检查环境
    print("检查环境...")
    test_env = AlphaReachEnv()
    check_env(test_env)
    test_env.close()
    print("环境检查通过")
    
    # 创建训练环境
    train_env = DummyVecEnv([make_env(i, log_path) for i in range(num_envs)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )

    # 创建评估环境
    eval_env = DummyVecEnv([make_env(0, log_path)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False
    )
    eval_env.obs_rms = train_env.obs_rms

    # ===== ✅ 核心修改:算法配置 =====
    configs = {
        'SAC': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'buffer_size': 500000,
            'batch_size': 512,
            'tau': 0.005,
            'gamma': 0.98,
            'train_freq': (max(1, 64 // num_envs), 'step'),
            'gradient_steps': 64,
            'learning_starts': 10000,
            'ent_coef': 'auto_0.2',
            'target_update_interval': 1,
            'use_sde': True,
            'sde_sample_freq': 4,
            'policy_kwargs': dict(net_arch=[256, 256, 256]),
            'verbose': 1
        },
        # ✅ TQC配置 - 针对水下环境优化
        'TQC': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'buffer_size': 500000,
            'batch_size': 512,
            'tau': 0.005,
            'gamma': 0.98,
            'train_freq': (max(1, 64 // num_envs), 'step'),
            'gradient_steps': 64,
            'learning_starts': 10000,
            'ent_coef': 'auto',  # TQC使用自动熵调整
            'target_update_interval': 1,
            'use_sde': True,
            'sde_sample_freq': 4,
            # TQC特有参数
            'top_quantiles_to_drop_per_net': 2,
            'n_quantiles': 25,
            'n_critics': 2,
            'policy_kwargs': dict(
                net_arch=[256, 256, 256],
                n_quantiles=25,
                n_critics=2
            ),
            'verbose': 1
        },
        # ✅ CrossQ配置 - 针对水下环境优化
        'CrossQ': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'buffer_size': 500000,
            'batch_size': 512,
            'tau': 0.005,
            'gamma': 0.98,
            'train_freq': (max(1, 64 // num_envs), 'step'),
            'gradient_steps': 64,
            'learning_starts': 10000,
            'ent_coef': 'auto',
            'target_update_interval': 1,
            'use_sde': True,
            'sde_sample_freq': 4,
            # CrossQ特有参数
            'n_critics': 2,
            'policy_kwargs': dict(
                net_arch=[256, 256, 256],
                n_critics=2
            ),
            'verbose': 1
        }
    }
    
    # 保存实验配置
    save_experiment_config(exp_dir, algorithm, configs[algorithm], total_timesteps)

    # ===== ✅ 创建模型 - 支持三种算法 =====
    algo_classes = {
        'SAC': SAC,
        'TQC': TQC,
        'CrossQ': CrossQ
    }
    
    if algorithm not in algo_classes:
        raise ValueError(f"不支持的算法: {algorithm}. 请选择 SAC/TQC/CrossQ")
    
    print(f"创建 {algorithm} 模型...")
    model = algo_classes[algorithm](
        env=train_env, 
        tensorboard_log=os.path.join(log_path, "tensorboard"),
        **configs[algorithm]
    )

    # ===== ✅ 关键修改:添加ComprehensiveMonitor =====
    print("\n初始化综合监控系统...")
    comprehensive_monitor = ComprehensiveMonitor(
        log_dir=log_path,
        verbose=1,
        window_size=100
    )
    
    # 设置回调函数
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, f"{algorithm}_best"),
        log_path=log_path,
        eval_freq=max(5000, total_timesteps // 20),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25000, total_timesteps // 4),
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix=f"{algorithm}_checkpoint"
    )

    progress_callback = TrainingProgressCallback(eval_freq=1000)
    
    # 开始训练
    print("开始训练...")
    start_time = time.time()
    
    # ✅ 关键:将comprehensive_monitor添加到callbacks列表
    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            eval_callback, 
            checkpoint_callback, 
            progress_callback, 
            comprehensive_monitor  # ✅ 添加综合监控
        ],
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"训练完成,耗时: {training_time:.1f}秒")

    # 保存最终模型
    model_path = os.path.join(save_path, f"{algorithm}_final")
    model.save(model_path)
    train_env.save(os.path.join(save_path, f"{algorithm}_vecnormalize.pkl"))

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

    # 保存训练总结
    training_summary = {
        'algorithm': algorithm,
        'total_timesteps': total_timesteps,
        'training_time': training_time,
        'final_mean_reward': float(mean_reward),
        'final_std_reward': float(std_reward),
        'experiment_name': exp_name,
        'model_path': model_path
    }

    with open(os.path.join(exp_dir, "training_summary.json"), 'w') as f:
        json.dump(training_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"实验完成!所有结果已保存到: {exp_dir}")
    print(f"{'='*60}")
    print(f"包含内容:")
    print(f"  - 模型权重: {os.path.join(exp_dir, 'models')}")
    print(f"  - 训练日志: {os.path.join(exp_dir, 'logs')}")
    print(f"  - ✅ 训练图表: {os.path.join(exp_dir, 'logs', 'plots')}")
    print(f"  - ✅ 训练指标: {os.path.join(exp_dir, 'logs', 'metrics')}")
    print(f"  - 实验配置: config.json")
    print(f"  - ✅ 训练报告: training_report.json")

    return model, mean_reward, std_reward, exp_dir


# ===== ✅ 测试函数 - 支持三种算法 =====
def test_trained_model(model_path, algorithm='SAC', n_episodes=5):
    """测试训练好的模型 - 支持SAC/TQC/CrossQ"""
    
    print(f"测试训练好的{algorithm}模型...")
    print(f"模型路径: {model_path}")
    
    env = AlphaReachEnv(render_mode="human")
    
    # ✅ 加载对应算法的模型
    algo_classes = {'SAC': SAC, 'TQC': TQC, 'CrossQ': CrossQ}
    
    if algorithm not in algo_classes:
        raise ValueError(f"不支持的算法: {algorithm}. 请选择 SAC/TQC/CrossQ")
    
    model = algo_classes[algorithm].load(model_path)
    
    # 加载VecNormalize
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
        
        for step in range(500):
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
        
        success = info.get('success', False)
        distance = info.get('distance', 0)
        
        if success:
            successes += 1
        
        total_rewards.append(episode_reward)
        distances.append(distance)
        
        print(f"Episode {episode+1}: 成功={success}, 距离={distance:.3f}m, "
              f"奖励={episode_reward:.2f}, 步数={step_count}")
    
    print("\n" + "="*50)
    print("测试结果统计:")
    print("="*50)
    print(f"成功率: {successes}/{n_episodes} ({successes/n_episodes*100:.1f}%)")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均最终距离: {np.mean(distances):.3f} ± {np.std(distances):.3f}m")
    
    env.close()
    
    return {
        'success_rate': successes / n_episodes,
        'mean_reward': np.mean(total_rewards),
        'mean_distance': np.mean(distances)
    }


# ===== ✅ 算法比较函数 =====
def compare_algorithms(algorithms=['SAC', 'TQC', 'CrossQ'], timesteps=300000):
    """比较不同算法的性能"""
    
    print("开始算法性能比较")
    print(f"算法: {algorithms}")
    print(f"训练步数: {timesteps}")
    print("="*50)
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\n正在训练 {algorithm}...")
        try:
            model, mean_reward, std_reward, _ = train_alpha_reach(
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
            import traceback
            traceback.print_exc()
            results[algorithm] = {
                'mean_reward': -np.inf,
                'std_reward': 0,
                'model': None
            }
    
    # 打印比较结果
    print("\n" + "="*60)
    print("算法性能比较结果:")
    print("="*60)
    print(f"{'算法':<10} {'平均奖励':<12} {'标准差':<10} {'相对性能':<10}")
    print("-" * 60)
    
    best_reward = max([r['mean_reward'] for r in results.values() if r['mean_reward'] != -np.inf])
    
    for algo, result in results.items():
        if result['mean_reward'] != -np.inf:
            relative_perf = result['mean_reward'] / best_reward * 100
            print(f"{algo:<10} {result['mean_reward']:<12.2f} {result['std_reward']:<10.2f} {relative_perf:<10.1f}%")
        else:
            print(f"{algo:<10} {'失败':<12} {'-':<10} {'-':<10}")
    
    best_algo = max(results.keys(), key=lambda x: results[x]['mean_reward'])
    if results[best_algo]['mean_reward'] != -np.inf:
        print(f"\n最佳算法: {best_algo} (奖励: {results[best_algo]['mean_reward']:.2f})")
    
    return results


# ===== ✅ 主函数 =====
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpha机械臂到达任务训练 - 支持SAC/TQC/CrossQ')
    parser.add_argument('--mode', choices=['train', 'test', 'compare'], 
                       default='train', help='运行模式')
    parser.add_argument('--algorithm', choices=['SAC', 'TQC', 'CrossQ'], 
                       default='SAC', help='RL算法')
    parser.add_argument('--timesteps', type=int, default=300000, 
                       help='训练步数')
    parser.add_argument('--num_envs', type=int, default=4,
                       help='并行环境数量')
    parser.add_argument('--model', type=str, 
                       help='测试模式下的模型路径')
    parser.add_argument('--episodes', type=int, default=5, 
                       help='测试回合数')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("单算法训练模式")
        train_alpha_reach(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            auto_visualize=False
        )
        
    elif args.mode == 'test':
        if not args.model:
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
            algorithms=['SAC', 'TQC', 'CrossQ'],
            timesteps=args.timesteps
        )


if __name__ == "__main__":
    main()