from stable_baselines3 import PPO
from envs.simple_env import PyBulletGymWrapper

def train():
    env = PyBulletGymWrapper(render=False)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_kuka")  # 生成 ppo_kuka.zip

if __name__ == "__main__":
    train()
