import os
import numpy as np
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# 导入你的环境
from envs.rl_env_v3_her import AlphaRobotHERFixed

def make_env():
    """创建环境"""
    def _init():
        env = AlphaRobotHERFixed(render_mode=None)  # 训练时不渲染
        return Monitor(env)
    return _init

def train_algorithm(algo_name, total_timesteps=50000):
    """训练单个算法"""
    print(f"\n训练 {algo_name}...")
    
    # 创建训练环境
    env = DummyVecEnv([make_env() for _ in range(1)])  # 1个环境
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 创建评估环境
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # 检查观察空间类型并选择正确的policy
    sample_env = AlphaRobotHERFixed(render_mode=None)
    is_dict_obs = hasattr(sample_env.observation_space, 'spaces')
    sample_env.close()
    
    if is_dict_obs:
        policy_type = 'MultiInputPolicy'
        print(f"检测到Dict观察空间，使用 {policy_type}")
    else:
        policy_type = 'MlpPolicy' 
        print(f"检测到Box观察空间，使用 {policy_type}")
    
    # 算法配置 (针对水下机器人优化)
    configs = {
        'PPO': {
            'policy': policy_type,
            'learning_rate': 3e-4,
            'n_steps': 512,
            'batch_size': 64,
            'gamma': 0.98,  # 稍低，因为任务相对简单
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01  # 适度探索
        },
        'SAC': {
            'policy': policy_type, 
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'batch_size': 256,
            'gamma': 0.98,
            'tau': 0.005,
            'ent_coef': 'auto'
        },
        'TD3': {
            'policy': policy_type,
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'batch_size': 256,
            'gamma': 0.98,
            'tau': 0.005,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5
        },
        'DDPG': {
            'policy': policy_type,
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'batch_size': 256,
            'gamma': 0.98,
            'tau': 0.005
        }
    }
    
    # 创建模型
    algo_classes = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3, 'DDPG': DDPG}
    model = algo_classes[algo_name](env=env, **configs[algo_name])
    
    # 设置回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{algo_name}_best",
        log_path=f"./logs/{algo_name}",
        eval_freq=2000,
        n_eval_episodes=5,
        deterministic=True
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"./models/{algo_name}_checkpoints"
    )
    
    # 训练
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # 保存最終模型
    model.save(f"./models/{algo_name}_final")
    env.save(f"./models/{algo_name}_vecnormalize.pkl")
    
    # 最終評估
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    
    print(f"{algo_name} 最終性能: {mean_reward:.2f} ± {std_reward:.2f}")
    
    env.close()
    eval_env.close()
    
    return mean_reward, std_reward

def compare_algorithms(algorithms=['PPO', 'SAC', 'TD3'], timesteps=50000):
    """比較多個算法"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    results = {}
    
    for algo in algorithms:
        mean_reward, std_reward = train_algorithm(algo, timesteps)
        results[algo] = {'mean': mean_reward, 'std': std_reward}
    
    # 打印比較結果
    print("\n" + "="*50)
    print("算法性能比較:")
    print("="*50)
    
    for algo, result in results.items():
        print(f"{algo:8s}: {result['mean']:7.2f} ± {result['std']:5.2f}")
    
    # 找出最佳算法
    best_algo = max(results.keys(), key=lambda x: results[x]['mean'])
    print(f"\n最佳算法: {best_algo}")
    
    return results

def test_trained_model(model_path, algo_name, n_episodes=5):
    """測試訓練好的模型"""
    # 創建測試環境 (帶渲染)
    env = AlphaRobotHERFixed(render_mode="human")
    
    # 加載模型
    algo_classes = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3, 'DDPG': DDPG}
    model = algo_classes[algo_name].load(model_path)
    
    print(f"\n測試 {algo_name} 模型...")
    
    successes = 0
    total_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(500):  # 最多500步
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
                
        total_rewards.append(episode_reward)
        if info.get('success', False):
            successes += 1
            
        print(f"Episode {episode+1}: 獎勵 {episode_reward:.2f}, 距離 {info.get('distance', 0):.3f}, 成功 {info.get('success', False)}")
    
    print(f"\n測試結果:")
    print(f"成功率: {successes}/{n_episodes} ({successes/n_episodes*100:.1f}%)")
    print(f"平均獎勵: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'compare', 'test'], default='compare')
    parser.add_argument('--algo', choices=['PPO', 'SAC', 'TD3', 'DDPG'], default='PPO')
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--model', type=str, help='模型路徑 (測試模式)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_algorithm(args.algo, args.timesteps)
    elif args.mode == 'compare':
        compare_algorithms(['PPO', 'SAC', 'TD3'], args.timesteps)
    elif args.mode == 'test':
        if not args.model:
            print("測試模式需要指定 --model 參數")
        else:
            test_trained_model(args.model, args.algo)