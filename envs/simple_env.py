import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

class SimpleRobotEnv:
    def __init__(self, render=False):
        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation)
        self.num_joints = p.getNumJoints(self.robot)
        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        joint_states = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        return np.array(joint_states)

    def step(self, action):
        for i in range(self.num_joints):
            current_pos = p.getJointState(self.robot, i)[0]
            target_pos = current_pos + action[i]
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=target_pos)
        p.stepSimulation()
        time.sleep(1. / 240.)
        obs = self._get_obs()
        reward = -np.linalg.norm(obs)
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


    def close(self):
        p.disconnect()

class PyBulletGymWrapper(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.env = SimpleRobotEnv(render)
        obs_dim = self.env.num_joints
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        pass

    def close(self):
        self.env.close()
