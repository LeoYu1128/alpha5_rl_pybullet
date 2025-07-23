# Alpha5 Underwater Manipulator RL Framework

**Reinforcement Learning for the Alpha5 Arm in PyBullet Simulation**
Using gymnasium and stablebaseline3<br>
run the train_new.py file <br>
e.g.<br>
python train_new.py --mode train --algo SAC --timesteps 50000 --n_envs 4<br>
(Support DDPG, SAC, PPO)<br>
Remember to use the virtual environment (venvs in this project).<br>
Current progress: Try to find a proper reward function.

## Quick Tutorial: Understanding How This Project Works

> If you're new here. This is a reinforcement learning project built for training the **Alpha5 underwater robotic arm** using **PyBullet** and **Stable-Baselines3**, with the environment wrapped in **Gymnasium** format.

### How Components Talk to Each Other

Here's the flow:

- **PyBullet**: A lightweight physics simulator that provides core rigid body dynamics, joint control, and collision detection.
- **Gymnasium**: A modern, well-maintained interface for reinforcement learning environments. It wraps PyBullet environments using the standard `reset()` and `step()` methods.
- **Stable-Baselines3**: A collection of reliable and scalable RL algorithms like PPO, SAC, TD3 — used to train your agent.

You write your own custom Gymnasium environment using PyBullet as the backend. Then you plug it directly into SB3 and start training!

### 🧪 Quickstart Training (TL;DR)

```bash
# Clone and install dependencies
git clone https://github.com/LeoYu1128/alpha5_train_by_pybullet.git
cd alpha5_train_by_pybullet
python -m venv venvs
source venvs/bin/activate
pip install -r requirements.txt

# Start training (with PPO for example)
python train_new.py --mode train --algo PPO

