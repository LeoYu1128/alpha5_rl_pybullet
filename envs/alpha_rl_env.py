"""
Alpha Robot 强化学习环境
用于PyBullet仿真的独立环境模块
"""

import pybullet as p
import pybullet_data
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AlphaRobotEnv(gym.Env):
    """Alpha Robot 强化学习环境"""
    
    def __init__(self, render_mode=None, max_steps=1e5, dense_reward=True):
        super(AlphaRobotEnv, self).__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.dense_reward = dense_reward
        
        # Alpha机械臂参数
        self.MAX_REACH = 0.4  # 最大可达范围 400mm
        self.MIN_REACH = 0.1  # 最小可达范围
        self.GRIPPER_RANGE = 0.04  # 夹爪最大开合范围
        
        # 关节配置 (根据实际Alpha机械臂)
        self.joint_limits = {
            'joint_1': (0.032, 6.02),      # axis_e - 旋转基座
            'joint_2': (0.0174533, 3.40339), # axis_d - 肩部
            'joint_3': (0.0174533, 3.40339), # axis_c - 肘部  
            'joint_4': (0.0, 3.22),          # axis_b - 腕部旋转（修正：不是无限旋转）
            'joint_5': (0.0013, 0.0133)      # axis_a - 夹爪
        }
        
        # 连接PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.2]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 初始化机器人和环境
        self._setup_scene()
        
        # 定义动作空间（5个关节的归一化动作）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(5,), 
            dtype=np.float32
        )
        
        # 定义观察空间
        # [关节位置(5), 关节速度(5), 末端位置(3), 末端姿态(4), 目标位置(3), 相对位置(3)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(23,), 
            dtype=np.float32
        )
        
        # 性能追踪
        self.episode_rewards = []
        self.success_count = 0
        
    def _setup_scene(self):
        """设置场景"""
        # 创建地面
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 加载机器人URDF
        self.robot_id = self._load_robot()
        
        # 获取关节索引
        self.joint_indices = []
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
        
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if joint_name in self.joint_names:
                self.joint_indices.append(i)
        
        # 获取末端执行器链接索引（TCP）
        for i in range(p.getNumJoints(self.robot_id)):
            link_info = p.getJointInfo(self.robot_id, i)
            if b'tcp' in link_info[12]:  # link name
                self.end_effector_index = i
                break
        else:
            # 如果没有tcp链接，使用最后一个链接
            self.end_effector_index = p.getNumJoints(self.robot_id) - 1
        
        # 创建目标球体
        self.target_visual = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.02, 
            rgbaColor=[1, 0, 0, 0.8]
        )
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.target_visual,
            basePosition=[0.2, 0.1, 0.2]
        )
        
        # 创建工作空间可视化（可选）
        if self.render_mode == "human":
            self._create_workspace_visualization()
            
    def _load_robot(self):
        # 加载机器人
        robot_id = p.loadURDF(
            "alpha_description/urdf/alpha_robot_test.urdf",
            basePosition=[0, 0, 0.1],
            useFixedBase=True
        )
        return robot_id
        
    def _create_workspace_visualization(self):
        """创建工作空间可视化"""
        # 绘制可达范围球体
        workspace_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.MAX_REACH,
            rgbaColor=[0.2, 0.2, 0.8, 0.1]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=workspace_visual,
            basePosition=[0, 0, 0.2]
        )
        
    def reset(self, seed=None, target_position=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_rewards = []
        
        # 重置机器人到初始位置
        initial_positions = [3.14, 1.57, 1.57, 1.57, 0.005]  # 中间位置
        for i, (joint_idx, pos) in enumerate(zip(self.joint_indices, initial_positions)):
            p.resetJointState(self.robot_id, joint_idx, pos)
        
        # 设置目标位置
        if target_position is None:
            self.target_position = self._sample_valid_target()
        else:
            self.target_position = np.array(target_position)
            
        p.resetBasePositionAndOrientation(
            self.target_id,
            self.target_position,
            [0, 0, 0, 1]
        )
        
        # 清除之前的距离记录
        if hasattr(self, 'prev_distance'):
            delattr(self, 'prev_distance')
        if hasattr(self, 'prev_end_effector_pos'):
            delattr(self, 'prev_end_effector_pos')
        # print("检查观测维数在rl_env的reset里面：", self.observation_space.shape)
        return self._get_observation(), {}
    
    def _sample_valid_target(self):
        """采样有效的目标位置"""
        while True:
            # 在球形工作空间内随机采样
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi/2)  # 只在上半球
            r = np.random.uniform(self.MIN_REACH, self.MAX_REACH)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi) + 0.1  # 加上基座高度偏移
            
            # 确保z坐标在合理范围内
            if 0.05 < z < 0.35:
                return np.array([x, y, z])
    
    def step(self, action):
        """执行动作"""
        self.current_step += 1
        
        # 将归一化动作映射到关节范围
        target_positions = []
        for i, joint_name in enumerate(self.joint_names):
            low, high = self.joint_limits[joint_name]
            target_pos = low + (action[i] + 1) * (high - low) / 2
            target_positions.append(target_pos)
        
        # 应用关节控制
        for joint_idx, target_pos in zip(self.joint_indices, target_positions):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=100,
                maxVelocity=1.0
            )
        
        # 步进仿真
        for _ in range(4):  # 4个子步骤
            p.stepSimulation()
            
        # 获取新状态
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._is_success()
        truncated = self.current_step >= self.max_steps
        
        # 记录
        self.episode_rewards.append(reward)
        if terminated:
            self.success_count += 1
            
        info = {
            'distance': self._get_distance_to_target(),
            'success': terminated,
            'episode_reward': sum(self.episode_rewards)
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """获取观察状态"""
        # 关节状态
        joint_positions = []
        joint_velocities = []
        
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])
        
        # 末端执行器状态
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        ee_pos = np.array(ee_state[0])
        ee_orn = np.array(ee_state[1])  # 四元数
        
        # 相对位置
        relative_pos = self.target_position - ee_pos
        
        # 组合观察
        observation = np.concatenate([
            joint_positions,      # 5
            joint_velocities,     # 5
            ee_pos,              # 3
            ee_orn,              # 4
            self.target_position, # 3
            relative_pos         # 3
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_reward(self):
        """计算奖励（支持稠密和稀疏奖励）"""
        ee_pos = self._get_end_effector_pos()
        distance = np.linalg.norm(ee_pos - self.target_position)
        
        if self.dense_reward:
            # 稠密奖励
            # 1. 距离奖励
            distance_reward = np.exp(-5 * distance)
            
            # 2. 进步奖励
            if hasattr(self, 'prev_distance'):
                progress = self.prev_distance - distance
                progress_reward = 10 * progress
            else:
                progress_reward = 0
            self.prev_distance = distance
            
            # 3. 成功奖励
            if distance < 0.02:
                success_reward = 100
            elif distance < 0.05:
                success_reward = 10
            else:
                success_reward = 0
                
            # 4. 平滑性惩罚
            joint_velocities = []
            for joint_idx in self.joint_indices:
                joint_state = p.getJointState(self.robot_id, joint_idx)
                joint_velocities.append(joint_state[1])
            smoothness_penalty = -0.01 * np.sum(np.square(joint_velocities))
            
            reward = distance_reward + progress_reward + success_reward + smoothness_penalty
            
        else:
            # 稀疏奖励
            if distance < 0.02:
                reward = 100
            else:
                reward = -1
                
        return reward
    
    def _get_end_effector_pos(self):
        """获取末端执行器位置"""
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        return np.array(ee_state[0])
    
    def _get_distance_to_target(self):
        """计算到目标的距离"""
        ee_pos = self._get_end_effector_pos()
        return np.linalg.norm(ee_pos - self.target_position)
    
    def _is_success(self):
        """检查是否成功到达目标"""
        return self._get_distance_to_target() < 0.02
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            # GUI模式自动渲染
            pass
            
    def close(self):
        """关闭环境"""
        if hasattr(self, 'physics_client'):
            p.disconnect(self.physics_client)
            
    def get_success_rate(self, window=100):
        """获取最近的成功率"""
        if hasattr(self, 'success_history'):
            recent = self.success_history[-window:]
            return sum(recent) / len(recent) if recent else 0
        return 0