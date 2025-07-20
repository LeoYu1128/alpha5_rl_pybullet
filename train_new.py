"""
使用Stable Baselines3训练和比较不同算法
"""

import os
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from envs.alpha_env import AlphaRobotEnv


class TensorboardCallback(BaseCallback):
    """自定义Tensorboard回调"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        
    def _on_step(self) -> bool:
        # 检查是否有episode结束
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                    self.episode_lengths.append(info['l'])
                    
                    # 记录成功率
                    if 'success' in info:
                        self.successes.append(info['success'])
                        
            # 记录到tensorboard
            if len(self.episode_rewards) > 0:
                self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards[-100:]))
                self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-100:]))
                
                if len(self.successes) > 0:
                    self.logger.record('rollout/success_rate', np.mean(self.successes[-100:]))
                    
        return True


def make_env(render_mode=None):
    """创建环境"""
    def _init():
        env = AlphaRobotEnv(render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init


def train_single_algorithm(algorithm_name, total_timesteps=100000, 
                          n_envs=4, save_dir="results", render=False):
    """训练单个算法"""
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algo_save_dir = os.path.join(save_dir, f"{algorithm_name}_{timestamp}")
    os.makedirs(algo_save_dir, exist_ok=True)
    
    # 创建向量化环境
    env = DummyVecEnv([make_env() for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 创建评估环境
    eval_env = DummyVecEnv([make_env(render_mode="human" if render else None)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, 
                            clip_obs=10., training=False)
    
    # 算法配置
    algorithm_configs = {
        'PPO': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'n_steps': 2048 // n_envs,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'use_sde': False,  # 可以尝试True以增加探索
            'sde_sample_freq': -1,
            'verbose': 1,
            'tensorboard_log': os.path.join(algo_save_dir, "tb_logs")
        },
        'SAC': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'buffer_size': 1000000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'ent_coef': 'auto',
            'target_update_interval': 1,
            'target_entropy': 'auto',
            'use_sde': False,
            'sde_sample_freq': -1,
            'use_sde_at_warmup': False,
            'verbose': 1,
            'tensorboard_log': os.path.join(algo_save_dir, "tb_logs")
        },
        'TD3': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'buffer_size': 1000000,
            'learning_starts': 1000,
            'batch_size': 100,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'action_noise': None,  # 使用参数噪声代替
            'optimize_memory_usage': False,
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            'verbose': 1,
            'tensorboard_log': os.path.join(algo_save_dir, "tb_logs")
        },
        'DDPG': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'buffer_size': 1000000,
            'learning_starts': 1000,
            'batch_size': 100,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'action_noise': None,
            'optimize_memory_usage': False,
            'verbose': 1,
            'tensorboard_log': os.path.join(algo_save_dir, "tb_logs")
        }
    }
    
    # 创建模型
    algorithm_class = {
        'PPO': PPO,
        'SAC': SAC,
        'TD3': TD3,
        'DDPG': DDPG
    }[algorithm_name]
    
    config = algorithm_configs[algorithm_name]
    model = algorithm_class(env=env, device="cpu", **config)

    
    # 创建回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(algo_save_dir, "best_model"),
        log_path=os.path.join(algo_save_dir, "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(algo_save_dir, "checkpoints"),
        name_prefix=algorithm_name
    )
    
    tensorboard_callback = TensorboardCallback()
    
    callbacks = CallbackList([eval_callback, checkpoint_callback, tensorboard_callback])
    
    # 训练
    print(f"\n{'='*50}")
    print(f"Training {algorithm_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Number of environments: {n_envs}")
    print(f"Save directory: {algo_save_dir}")
    print(f"{'='*50}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        tb_log_name=algorithm_name,
        reset_num_timesteps=True,
        progress_bar=True
    )
    
    # 保存最终模型和环境
    model.save(os.path.join(algo_save_dir, "final_model"))
    env.save(os.path.join(algo_save_dir, "vec_normalize.pkl"))
    
    # 最终评估
    print(f"\nFinal evaluation of {algorithm_name}...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # 清理
    env.close()
    eval_env.close()
    
    return mean_reward, std_reward, algo_save_dir


def compare_algorithms(algorithms=['PPO', 'SAC', 'TD3'], 
                      total_timesteps=100000,
                      n_seeds=3):
    """比较多个算法"""
    
    results = {}
    
    for algo in algorithms:
        algo_results = {
            'mean_rewards': [],
            'std_rewards': [],
            'save_dirs': []
        }
        
        for seed in range(n_seeds):
            print(f"\n\n{'#'*60}")
            print(f"Training {algo} with seed {seed}")
            print(f"{'#'*60}\n")
            
            # 设置随机种子
            np.random.seed(seed)
            
            # 训练
            mean_reward, std_reward, save_dir = train_single_algorithm(
                algo,
                total_timesteps=total_timesteps,
                save_dir=f"results/comparison/seed_{seed}"
            )
            
            algo_results['mean_rewards'].append(mean_reward)
            algo_results['std_rewards'].append(std_reward)
            algo_results['save_dirs'].append(save_dir)
            
        results[algo] = algo_results
        
    # 打印比较结果
    print("\n\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    for algo, res in results.items():
        mean_of_means = np.mean(res['mean_rewards'])
        std_of_means = np.std(res['mean_rewards'])
        
        print(f"\n{algo}:")
        print(f"  Average performance: {mean_of_means:.2f} +/- {std_of_means:.2f}")
        print(f"  Individual runs: {res['mean_rewards']}")
        
    print("\n" + "="*60)
    
    return results


def test_trained_model(model_path, env_path=None, n_episodes=10, render=True):
    """测试训练好的模型"""
    
    # 创建环境
    env = AlphaRobotEnv(render_mode="human" if render else None)
    
    # 如果有保存的归一化参数，加载它们
    if env_path and os.path.exists(env_path):
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(env_path, env)
        env.training = False
        env.norm_reward = False
    
    # 加载模型
    # 自动检测模型类型
    for algo_class in [PPO, SAC, TD3, DDPG]:
        try:
            model = algo_class.load(model_path)
            print(f"Loaded {algo_class.__name__} model")
            break
        except:
            continue
    else:
        raise ValueError("Could not load model")
    
    # 测试
    print(f"\nTesting model: {model_path}")
    print(f"Number of episodes: {n_episodes}")
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
                
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 处理向量化环境的info
        if isinstance(info, list):
            info = info[0]
        successes.append(info.get('success', False))
        
        print(f"Episode {episode + 1}: "
              f"Reward = {episode_reward:.2f}, "
              f"Length = {episode_length}, "
              f"Success = {info.get('success', False)}")
    
    # 统计
    print(f"\n{'='*40}")
    print("Test Results:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f}")
    print(f"Success Rate: {np.mean(successes):.2%}")
    print(f"{'='*40}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'compare', 'test'], default='train')
    parser.add_argument('--algo', choices=['PPO', 'SAC', 'TD3', 'DDPG'], default='PPO')
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--model_path', type=str, help='Path to trained model (for test mode)')
    parser.add_argument('--env_path', type=str, help='Path to VecNormalize stats (for test mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_single_algorithm(
            args.algo,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs
        )
    elif args.mode == 'compare':
        compare_algorithms(
            algorithms=['PPO', 'SAC', 'TD3'],
            total_timesteps=args.timesteps
        )
    elif args.mode == 'test':
        if not args.model_path:
            raise ValueError("--model_path required for test mode")
        test_trained_model(args.model_path, args.env_path)