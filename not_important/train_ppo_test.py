import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.alpha_rl_env import AlphaRobotEnv

# 环境创建函数
def make_env_fn(seed=0):
    def _init():
        env = AlphaRobotEnv()
        env = Monitor(env)  # 添加监控器用于日志记录
        env.seed(seed)
        return env
    return _init

# 创建并行环境
n_envs = 4
env = SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])
eval_env = DummyVecEnv([make_env_fn(999)])  # 单个环境用于评估

# 日志和保存路径
log_dir = "./ppo_alpha_logs/"
os.makedirs(log_dir, exist_ok=True)
model_path = os.path.join(log_dir, "best_model")

# 回调设置：当平均 reward 达到阈值时停止训练
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    best_model_save_path=model_path,
    log_path=log_dir,
    eval_freq=5000,
    deterministic=True,
    render=False
)

# 创建 PPO 模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=os.path.join(log_dir, "tb"),
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.01,
    learning_rate=2.5e-4,
    vf_coef=0.5,
    max_grad_norm=0.5
)

# 开始训练
model.learn(
    total_timesteps=300_000,
    callback=eval_callback
)

# 加载最佳模型并评估
model = PPO.load(os.path.join(model_path, "best_model"))

obs, _ = eval_env.reset()
done = False
total_reward = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = eval_env.step(action)
    total_reward += reward
print("Best model evaluation reward:", total_reward)
