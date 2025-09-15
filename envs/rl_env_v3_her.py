# 修复PyBullet连接问题的版本
import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

class AlphaRobotHERFixed(gym.Env):
    """修复PyBullet连接问题的HER环境"""
    
    def __init__(self, render_mode="human", max_steps=3000):
        super().__init__()
        self.height_offset = 0.1
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.physics_client = None  # 重要：初始化为None
        
        # 保持原始配置
        self.joint_limits = {
            'lower': np.array([0.0, -3.49, 0.0, 0.0]),       
            'upper': np.array([6.10, 3.49, 3.22, 3.22]),
            'max_torque': np.array([9.0, 9.0, 9.0, 9.0])
        }
        
        # 延迟连接PyBullet
        self._connect_physics()
        self._setup_scene()
        
        # HER兼容空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        obs_dim = 14
        goal_dim = 3
        
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32)
        })
        
        print(f"Alpha Robot HER环境初始化完成 (修复版)")
    
    def _connect_physics(self):
        """安全连接PyBullet"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
        
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.physics_client)
            p.resetDebugVisualizerCamera(1.0, 45, -30, [0, 0, 0.3], physicsClientId=self.physics_client)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
    
    def _setup_scene(self):
        """设置场景"""
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        
        # 加载机器人
        robot_paths = [
            "alpha_description/urdf/alpha_robot_for_pybullet.urdf",
            "../alpha_description/urdf/alpha_robot_for_pybullet.urdf", 
            "alpha_robot_for_pybullet.urdf"
        ]
        
        self.robot_id = None
        for robot_path in robot_paths:
            if os.path.exists(robot_path):
                try:
                    self.robot_id = p.loadURDF(
                        robot_path,
                        basePosition=[0, 0, self.height_offset],
                        useFixedBase=True,
                        physicsClientId=self.physics_client
                    )
                    print(f"成功加载机器人URDF: {robot_path}")
                    break
                except Exception as e:
                    continue
        
        if self.robot_id is None:
            raise FileNotFoundError("找不到Alpha机器人URDF文件")
        
        self.joint_indices = [2, 3, 4, 5]
        self.tcp_index = 11
        
        # 设置阻尼
        for joint_idx in self.joint_indices:
            p.changeDynamics(
                self.robot_id, joint_idx, 
                jointDamping=0.5,
                physicsClientId=self.physics_client
            )
    
    def _sample_goal(self):
        """简单直接的版本"""
        max_attempts = 50
        
        for _ in range(max_attempts):
            # 直接在立方体内随机采样
            x = np.random.uniform(-0.3, 0.3)
            y = np.random.uniform(-0.3, 0.3) 
            z = np.random.uniform(0.05, 0.3)  # 相对于地面的绝对高度
            
            goal = np.array([x, y, z])
            
            # 检查是否在以(0, 0, self.height_offset)为中心的0.4m球内
            distance = np.linalg.norm(goal - [0, 0, self.height_offset])
            
            if distance <= 0.39:  # 留1cm安全余量
                return goal.astype(np.float32)
        
        # fallback
        return np.array([0.2, 0.0, 0.2 + self.height_offset], dtype=np.float32)
    
    def _get_end_effector_pos(self):
        """获取末端位置"""
        if self.physics_client is None:
            return np.zeros(3, dtype=np.float32)
        
        link_state = p.getLinkState(
            self.robot_id, self.tcp_index, 
            physicsClientId=self.physics_client
        )
        return np.array(link_state[0], dtype=np.float32)
        
    def _get_joint_positions(self):
        """获取关节位置"""
        if self.physics_client is None:
            return np.zeros(4, dtype=np.float32)
        
        positions = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(
                self.robot_id, joint_idx, 
                physicsClientId=self.physics_client
            )
            positions.append(joint_state[0])
        return np.array(positions, dtype=np.float32)
        
    def _get_joint_velocities(self):
        """获取关节速度"""
        if self.physics_client is None:
            return np.zeros(4, dtype=np.float32)
        
        velocities = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(
                self.robot_id, joint_idx, 
                physicsClientId=self.physics_client
            )
            velocities.append(joint_state[1])
        return np.array(velocities, dtype=np.float32)
    
    def _get_obs(self):
        """获取观察"""
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        ee_pos = self._get_end_effector_pos()
        
        observation = np.concatenate([
            joint_positions,
            joint_velocities,
            ee_pos,
            self.goal
        ]).astype(np.float32)
        
        return {
            'observation': observation,
            'achieved_goal': ee_pos.copy(),
            'desired_goal': self.goal.copy()
        }
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """HER奖励函数"""
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        success = (distance < 0.03).astype(np.float32)
        return success - 1.0
    
    def _is_success(self, achieved_goal, desired_goal):
        """成功判断"""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance < 0.03
    
    def step(self, action):
        """环境步进"""
        if self.physics_client is None:
            self._connect_physics()
            self._setup_scene()
        
        self.current_step += 1
        
        # 位置控制
        current_pos = self._get_joint_positions()
        target_pos = current_pos + action * 0.05
        target_pos = np.clip(target_pos, 
                           self.joint_limits['lower'], 
                           self.joint_limits['upper'])
        
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos[i],
                maxVelocity=1.0,
                physicsClientId=self.physics_client
            )
            
        # 仿真
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.physics_client)
            
        obs = self._get_obs()
        
        reward = self.compute_reward(
            obs['achieved_goal'], 
            obs['desired_goal'], 
            None
        )
        
        success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        terminated = success
        truncated = self.current_step >= self.max_steps
        
        info = {
            'is_success': success,
            'distance': np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        }
        
        return obs, reward, terminated, truncated, info
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 确保连接存在
        if self.physics_client is None:
            self._connect_physics()
            self._setup_scene()
        
        self.current_step = 0
        
        # 重置关节
        init_positions = [
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-0.5, 1.0),
            np.random.uniform(-0.5, 1.0)
        ]
        
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(
                self.robot_id, joint_idx, init_positions[i],
                physicsClientId=self.physics_client
            )
            
        # 新目标
        self.goal = self._sample_goal()
        
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.physics_client)
            
        # 目标可视化
        if hasattr(self, 'target_id'):
            try:
                p.removeBody(self.target_id, physicsClientId=self.physics_client)
            except:
                pass
        
        target_visual = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.02, 
            rgbaColor=[1, 0, 0, 0.8],
            physicsClientId=self.physics_client
        )
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=self.goal,
            physicsClientId=self.physics_client
        )
        
        obs = self._get_obs()
        info = {'distance': np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])}
        
        target_distance = np.linalg.norm(self.goal)
        print(f"重置完成 - 新目标: {self.goal}, 距离: {target_distance*1000:.0f}mm")
        
        return obs, info
        
    def close(self):
        """关闭环境"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
            self.physics_client = None

# 简化的训练脚本
def simple_her_train():
    """简化训练，只用SAC+HER"""
    from stable_baselines3 import SAC
    from stable_baselines3.her import HerReplayBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    print("创建HER训练环境...")
    
    # 只创建一个环境，避免多连接问题
    env = DummyVecEnv([lambda: AlphaRobotHERFixed(render_mode=None)])
    
    model = SAC(
        'MultiInputPolicy',
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            'n_sampled_goal': 4,
            'goal_selection_strategy': 'future',
        },
        learning_rate=3e-4,
        batch_size=256,
        learning_starts=1000,
        verbose=1
    )
    
    print("开始SAC+HER训练...")
    model.learn(total_timesteps=50_000)
    model.save("alpha_robot_her_fixed")
    
    print("训练完成！开始测试...")
    
    # 测试
    test_env = AlphaRobotHERFixed(render_mode="human")
    obs, _ = test_env.reset()
    
    successes = 0
    episodes = 0
    
    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        if terminated or truncated:
            episodes += 1
            if info['is_success']:
                successes += 1
            print(f"Episode {episodes}: 成功={info['is_success']}, 距离={info['distance']:.3f}m")
            obs, _ = test_env.reset()
            
            if episodes >= 10:
                break
    
    print(f"最终成功率: {successes}/{episodes} = {successes/episodes*100:.1f}%")
    test_env.close()

if __name__ == "__main__":
    simple_her_train()