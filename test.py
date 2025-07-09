import gymnasium as gym
from stable_baselines3 import PPO
from envs.simple_env import PyBulletGymWrapper

def test():
    env = PyBulletGymWrapper(render=True)
    model = PPO.load("ppo_kuka")
    obs, info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, info = env.reset()
    env.close()

if __name__ == "__main__":
    test()
