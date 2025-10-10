import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

class AlphaReachEnv(gym.Env):
    """
    水下Alpha机械臂到达任务环境（带简单速度估计，无夹爪控制）
    
    关键设计：
    1. 模型只学习4个关节的到达任务
    2. 夹爪始终保持张开，不在强化学习范畴
    3. 使用真实角度训练（编码器读数）
    4. 目标随水流漂移，最大5cm
    5. 包含简单速度估计（从历史位置计算）
    """
    
    def __init__(self, render_mode=None, max_steps=800, reward_type='dense'):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.physics_client = None
        self.reward_type = reward_type
        
        # ✅ 目标漂移参数
        self.max_target_drift = 0.04  # 5cm最大漂移
        self.drift_damping = 0.95
        self.drift_noise_strength = 0.01
        self.target_velocity = np.zeros(3, dtype=np.float32)
        
        # ✅ 速度估计参数（历史位置法）
        self.target_position_history = []
        self.history_length = 3
        self.dt = 1./240.

        # 真实机械臂的home位置(编码器读数)
        self.real_home_positions = np.array([
            np.radians(2.34),
            np.radians(87.8),   # 真实87.8°时直立
            np.radians(1.0),
            np.radians(0.1)
        ])
        
        # URDF内部表示(仅用于PyBullet)
        self._urdf_home_positions = np.array([
            np.radians(2.34),
            np.radians(0),      # URDF 0°表示直立
            np.radians(1.0),
            np.radians(0.1)
        ])
        
        # 转换偏移(仅内部使用)
        # self._angle_offset = np.array([0, np.radians(87.8), 0, 0]) //////////////////////////////////////////////////////////////////////////
        self._angle_offset = np.array([0, 0, 0, 0])
        # 夹爪home位置(完全张开)
        self.home_gripper_position = 0.0133
        
        # 水下环境参数
        self.water_density = 1000.0
        self.gravity = -9.81
        self.base_height = 0.1
        self.workspace_radius = 0.4
        
        # 真实机械臂质量参数
        self.urdf_total_mass = 1.52
        self.target_actual_mass = 1.36
        self.target_underwater_mass = 0.9
        
        buoyancy_mass = self.target_actual_mass - self.target_underwater_mass
        self.buoyancy_compensation_ratio = buoyancy_mass / self.target_actual_mass
        self.robot_mass_scale = self.target_actual_mass / self.urdf_total_mass
        
        # 流体动力学参数
        self.drag_coefficient = 0.5
        self.buoyancy_enabled = True
        self.current_variation = True
        self.turbulence_strength = 0.015
        
        print("\n" + "="*70)
        print("水下Alpha机械臂环境 - 带速度估计（无夹爪控制）")
        print("="*70)
        print(f"真实Home位置: {np.degrees(self.real_home_positions)}")
        print(f"目标最大漂移: {self.max_target_drift*100:.1f}cm")
        print("速度估计方法: 简单差分（从历史位置）")
        print("夹爪: 始终保持张开，不参与训练")
        print("="*70 + "\n")

        # 连接物理引擎
        self._connect_physics()
        self._setup_underwater_scene()
        self._analyze_robot()

        # 奖励参数
        self.distance_scale = 1.0
        self.progress_scale = 1.2
        self.control_penalty = 0.01
        self.velocity_penalty = 0.02
        self.time_penalty = 0.003
        self.success_bonus = 30.0
        self.success_threshold = 0.02
        self.previous_distance = None

        # ✅ 动作空间: 只有4个关节（无夹爪）
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            shape=(4,),
            dtype=np.float32
        )
        
        # ✅ 观察空间: 17维（4关节位置 + 4关节速度 + 3末端位置 + 3目标位置 + 3目标速度估计）
        obs_dim = 4 + 4 + 3 + 3 + 3  # 17维
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        print(f"✅ 动作空间: {self.action_space.shape} (4个关节)")
        print(f"✅ 观察空间: {self.observation_space.shape} (含目标速度估计)")
    
    def _real_to_urdf(self, real_angles):
        """内部: 真实角度 -> URDF角度"""
        return np.array(real_angles) - self._angle_offset
    
    def _urdf_to_real(self, urdf_angles):
        """内部: URDF角度 -> 真实角度"""
        return np.array(urdf_angles) + self._angle_offset
    
    def _connect_physics(self):
        """连接PyBullet"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
        
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, 
                                      physicsClientId=self.physics_client)
            p.resetDebugVisualizerCamera(2.0, 30, -20, [0, 0, 0.3], 
                                         physicsClientId=self.physics_client)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), 
                                  physicsClientId=self.physics_client)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.physics_client)
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
    
    def _setup_underwater_scene(self):
        """设置水下场景"""
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.1, 0.2, 0.4, 1.0],
                           physicsClientId=self.physics_client)
        
        # 加载机械臂
        robot_paths = [
            "alpha_robot_for_pybullet.urdf",
            "alpha_description/urdf/alpha_robot_for_pybullet.urdf",
            "../alpha_description/urdf/alpha_robot_for_pybullet.urdf",
        ]
        
        self.robot_id = None
        for robot_path in robot_paths:
            if os.path.exists(robot_path):
                try:
                    self.robot_id = p.loadURDF(
                        robot_path,
                        basePosition=[0, 0, self.base_height],
                        useFixedBase=True,
                        flags=p.URDF_USE_SELF_COLLISION,
                        physicsClientId=self.physics_client
                    )
                    
                    # 设置到home位置
                    num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
                    for i in range(num_joints):
                        joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
                        joint_name = joint_info[1].decode('utf-8')
                        
                        if joint_name == 'joint_1':
                            p.resetJointState(self.robot_id, i, self._urdf_home_positions[0],
                                            physicsClientId=self.physics_client)
                        elif joint_name == 'joint_2':
                            p.resetJointState(self.robot_id, i, self._urdf_home_positions[1],
                                            physicsClientId=self.physics_client)
                        elif joint_name == 'joint_3':
                            p.resetJointState(self.robot_id, i, self._urdf_home_positions[2],
                                            physicsClientId=self.physics_client)
                        elif joint_name == 'joint_4':
                            p.resetJointState(self.robot_id, i, self._urdf_home_positions[3],
                                            physicsClientId=self.physics_client)
                        elif joint_name == 'joint_5':
                            # 夹爪始终张开
                            p.resetJointState(self.robot_id, i, self.home_gripper_position,
                                            physicsClientId=self.physics_client)
                    
                    for _ in range(100):
                        p.stepSimulation(physicsClientId=self.physics_client)
                    
                    print(f"✅ 加载URDF: {robot_path}")
                    break
                    
                except Exception as e:
                    continue
        
        if self.robot_id is None:
            raise FileNotFoundError("找不到Alpha机械臂URDF文件")
    
    def _analyze_robot(self):
        """分析机器人结构"""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        self.joint_info = {}
        self.main_joint_indices = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            self.joint_info[i] = {
                'name': joint_name,
                'type': joint_type,
                'lower': lower_limit,
                'upper': upper_limit
            }
            
            if joint_name in ['joint_1', 'joint_2', 'joint_3', 'joint_4']:
                self.main_joint_indices.append(i)
        
        # 找末端执行器
        self.tcp_index = None
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            if 'tcp' in joint_name.lower():
                self.tcp_index = i
                break
        
        if self.tcp_index is None:
            self.tcp_index = num_joints - 1
        
        self._setup_underwater_dynamics()
    
    def _setup_underwater_dynamics(self):
        """设置水下动力学"""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        self.link_masses = {}
        self.link_indices = []
        
        for i in range(-1, num_joints):
            try:
                dynamics_info = p.getDynamicsInfo(self.robot_id, i, physicsClientId=self.physics_client)
                urdf_mass = dynamics_info[0]
                
                if urdf_mass > 0:
                    real_mass = urdf_mass * self.robot_mass_scale
                    self.link_masses[i] = real_mass
                    self.link_indices.append(i)
                    p.changeDynamics(self.robot_id, i, mass=real_mass,
                                   physicsClientId=self.physics_client)
                
                if i >= 0:
                    p.changeDynamics(self.robot_id, i,
                                   linearDamping=2.0, angularDamping=2.0, jointDamping=0.5,
                                   physicsClientId=self.physics_client)
                else:
                    p.changeDynamics(self.robot_id, i,
                                   linearDamping=1.0, angularDamping=1.0,
                                   physicsClientId=self.physics_client)
            except:
                pass
    
    # ✅ ===== 核心：简单速度估计 =====
    def _estimate_target_velocity(self):
        """
        从历史位置估计目标速度（现实可部署）
        """
        if len(self.target_position_history) < 2:
            return np.zeros(3, dtype=np.float32)
        
        # 简单差分
        current_pos = self.target_position_history[-1]
        previous_pos = self.target_position_history[-2]
        estimated_velocity = (current_pos - previous_pos) / self.dt
        
        # 多帧平滑（如果有3帧以上）
        if len(self.target_position_history) >= 3:
            velocities = []
            for i in range(1, min(self.history_length, len(self.target_position_history))):
                v = (self.target_position_history[-i] - 
                     self.target_position_history[-(i+1)]) / self.dt
                velocities.append(v)
            estimated_velocity = np.mean(velocities, axis=0)
        
        # 限制最大速度
        max_velocity = 0.15
        velocity_magnitude = np.linalg.norm(estimated_velocity)
        if velocity_magnitude > max_velocity:
            estimated_velocity *= (max_velocity / velocity_magnitude)
        
        return estimated_velocity.astype(np.float32)
    
    def _update_target_position(self, dt=1./240.):
        """更新目标位置 - 随机摆动（像浮标）"""
        if not hasattr(self, 'target_position'):
            return
        
        # 1. 恢复力
        displacement = self.target_position - self.initial_target_position
        distance_from_initial = np.linalg.norm(displacement)
        
        if distance_from_initial > 0.001:
            restore_strength = min(distance_from_initial / self.max_target_drift, 1.0) ** 2
            restore_force = -displacement * restore_strength * 0.5
        else:
            restore_force = np.zeros(3)
        
        # 2. 随机扰动
        random_current = np.array([
            np.random.uniform(-self.drift_noise_strength, self.drift_noise_strength),
            np.random.uniform(-self.drift_noise_strength, self.drift_noise_strength),
            np.random.uniform(-self.drift_noise_strength * 0.5, self.drift_noise_strength * 0.5)
        ])
        
        # 3. 周期性波浪
        time_factor = self.current_step * 0.01
        wave_force = np.array([
            0.005 * np.sin(time_factor * 2.0),
            0.005 * np.cos(time_factor * 1.5),
            0.003 * np.sin(time_factor * 0.8)
        ])
        
        # 4. 更新速度
        total_force = restore_force + random_current + wave_force
        self.target_velocity += total_force
        self.target_velocity *= self.drift_damping
        
        # 限制最大速度
        max_velocity = 0.02
        velocity_magnitude = np.linalg.norm(self.target_velocity)
        if velocity_magnitude > max_velocity:
            self.target_velocity *= (max_velocity / velocity_magnitude)
        
        # 5. 更新位置
        self.target_position += self.target_velocity * dt
        
        # 6. 硬性限制：不超过5cm
        displacement = self.target_position - self.initial_target_position
        distance_from_initial = np.linalg.norm(displacement)
        
        if distance_from_initial > self.max_target_drift:
            direction = displacement / distance_from_initial
            self.target_position = self.initial_target_position + direction * self.max_target_drift
            self.target_velocity = -self.target_velocity * 0.3
        
        # 7. 工作空间限制
        x, y, z = self.target_position
        r_xy = np.sqrt(x**2 + y**2)
        if r_xy > self.workspace_radius * 0.9:
            scale = (self.workspace_radius * 0.9) / r_xy
            x *= scale
            y *= scale
            self.target_velocity[:2] *= scale
        
        z_min = self.base_height + 0.05
        z_max = 0.35 + self.base_height
        if z < z_min:
            z = z_min
            self.target_velocity[2] = abs(self.target_velocity[2]) * 0.3
        elif z > z_max:
            z = z_max
            self.target_velocity[2] = -abs(self.target_velocity[2]) * 0.3
        
        self.target_position = np.array([x, y, z], dtype=np.float32)
    
    def _update_current_velocity(self):
        """更新水流速度"""
        if self.current_variation:
            time_factor = self.current_step * 0.01
            base_current = self.current_velocity.copy()
            periodic_variation = np.array([
                0.01 * np.sin(time_factor),
                0.01 * np.cos(time_factor * 1.5),
                0.01 * np.sin(time_factor * 0.5)
            ])
            turbulence = np.random.normal(0, self.turbulence_strength, 3)
            self.current_velocity_actual = base_current + periodic_variation + turbulence
        else:
            self.current_velocity_actual = self.current_velocity.copy()
    
    def _apply_underwater_forces(self):
        """应用水下力"""
        self._update_current_velocity()
        
        # 浮力
        if self.buoyancy_enabled:
            for link_idx in self.link_indices:
                try:
                    mass = self.link_masses.get(link_idx, 0)
                    if mass > 0:
                        gravity_force = mass * abs(self.gravity)
                        buoyancy_force = gravity_force * self.buoyancy_compensation_ratio
                        p.applyExternalForce(
                            self.robot_id, link_idx,
                            forceObj=[0, 0, buoyancy_force],
                            posObj=[0, 0, 0],
                            flags=p.LINK_FRAME,
                            physicsClientId=self.physics_client
                        )
                except:
                    pass
        
        # 流体阻力
        key_links = [self.tcp_index] + self.main_joint_indices[:3]
        
        for link_idx in key_links:
            try:
                link_state = p.getLinkState(self.robot_id, link_idx, computeLinkVelocity=1,
                                           physicsClientId=self.physics_client)
                link_velocity = np.array(link_state[6])
                relative_velocity = link_velocity - self.current_velocity_actual
                speed = np.linalg.norm(relative_velocity)
                
                if speed > 0.001:
                    characteristic_area = 0.008 if link_idx == self.tcp_index else 0.015
                    drag_magnitude = 0.5 * self.water_density * self.drag_coefficient * \
                                    characteristic_area * speed ** 2
                    
                    if speed > 1e-6:
                        drag_direction = -relative_velocity / speed
                        drag_force = drag_direction * drag_magnitude
                    else:
                        drag_force = np.zeros(3)
                    
                    drag_force = np.clip(drag_force, -12.0, 12.0)
                    p.applyExternalForce(self.robot_id, link_idx, forceObj=drag_force.tolist(),
                                       posObj=[0, 0, 0], flags=p.LINK_FRAME,
                                       physicsClientId=self.physics_client)
            except:
                pass
    
    def _get_joint_positions_real(self):
        """获取关节位置（真实角度）"""
        urdf_positions = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx,
                                         physicsClientId=self.physics_client)
            urdf_positions.append(joint_state[0])
        
        urdf_positions = np.array(urdf_positions, dtype=np.float32)
        real_positions = self._urdf_to_real(urdf_positions)
        return real_positions
    
    def _get_joint_velocities(self):
        """获取关节速度"""
        velocities = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx,
                                         physicsClientId=self.physics_client)
            velocities.append(joint_state[1])
        return np.array(velocities, dtype=np.float32)
    
    def _get_end_effector_position(self):
        """获取末端执行器位置"""
        link_state = p.getLinkState(self.robot_id, self.tcp_index,
                                    physicsClientId=self.physics_client)
        return np.array(link_state[0], dtype=np.float32)
    
    # ✅ ===== 核心：构建观察空间（含速度估计）=====
    def _get_observation(self):
        """
        17维观察: [4关节真实角度, 4关节速度, 3末端位置, 3目标位置, 3目标速度估计]
        """
        joint_positions_real = self._get_joint_positions_real()  # 真实角度
        joint_velocities = self._get_joint_velocities()
        ee_position = self._get_end_effector_position()
        
        # ✅ 估计目标速度
        target_velocity_estimated = self._estimate_target_velocity()
        
        observation = np.concatenate([
            joint_positions_real,      # 4维（真实角度）
            joint_velocities,          # 4维
            ee_position,               # 3维
            self.target_position,      # 3维
            target_velocity_estimated  # 3维（估计的）
        ]).astype(np.float32)
        
        return observation
    
    def _apply_action(self, action):
        """应用动作（只控制4个关节）"""
        current_real = self._get_joint_positions_real()
        
        scaled_action = action * 0.1
        target_real = current_real + scaled_action
        
        target_urdf = self._real_to_urdf(target_real)
        
        # 应用关节限制
        for i, joint_idx in enumerate(self.main_joint_indices):
            joint = self.joint_info[joint_idx]
            target_urdf[i] = np.clip(target_urdf[i], joint['lower'], joint['upper'])
        
        # 控制PyBullet
        control_torque = 27.0
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.setJointMotorControl2(
                self.robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=target_urdf[i],
                maxVelocity=0.5,
                force=control_torque,
                physicsClientId=self.physics_client
            )
        
        # ✅ 夹爪始终保持张开（不参与训练）
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            if joint_info[1].decode('utf-8') == 'joint_5':
                p.setJointMotorControl2(
                    self.robot_id, i, p.POSITION_CONTROL,
                    targetPosition=self.home_gripper_position,
                    force=10,
                    physicsClientId=self.physics_client
                )
                break
    
    def _compute_reward(self, action):
        """计算奖励"""
        achieved_goal = self._get_end_effector_position()
        desired_goal = self.target_position
        
        distance = float(np.linalg.norm(achieved_goal - desired_goal))
        prev_distance = self.previous_distance if self.previous_distance is not None else distance
        distance_delta = prev_distance - distance
        
        shaped_distance = -self.distance_scale * distance
        progress_reward = self.progress_scale * distance_delta
        control_cost = self.control_penalty * float(np.linalg.norm(action) ** 2)
        velocity_cost = self.velocity_penalty * float(np.linalg.norm(self._get_joint_velocities()))
        time_cost = self.time_penalty
        
        success = distance < self.success_threshold
        success_bonus = self.success_bonus if success else 0.0
        total_reward = shaped_distance + progress_reward - control_cost - velocity_cost - time_cost + success_bonus
        
        self.previous_distance = distance
        
        reward_terms = {
            'reward_distance': shaped_distance,
            'reward_progress': progress_reward,
            'reward_control': -control_cost,
            'reward_velocity': -velocity_cost,
            'reward_time': -time_cost,
            'reward_success': success_bonus
        }
        
        return float(total_reward), bool(success), reward_terms
    
    def _sample_target_position(self):
        """采样目标位置"""
        safe_radius_min = 0.12
        safe_radius_max = 0.35
        z_min = self.base_height
        z_max = 0.35 + self.base_height
        
        r = np.random.uniform(safe_radius_min, safe_radius_max)
        theta = np.random.uniform(-np.pi/2, np.pi/2)
        z = np.random.uniform(z_min, z_max)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return np.array([x, y, z], dtype=np.float32)
    
    def _update_target_visual(self):
        """更新目标视觉"""
        if hasattr(self, 'target_visual_id') and self.target_visual_id is not None:
            try:
                p.resetBasePositionAndOrientation(
                    self.target_visual_id, self.target_position, [0, 0, 0, 1],
                    physicsClientId=self.physics_client
                )
            except:
                pass
    
    def _create_target_visual(self):
        """创建目标视觉"""
        if hasattr(self, 'target_visual_id'):
            try:
                p.removeBody(self.target_visual_id, physicsClientId=self.physics_client)
            except:
                pass
        
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.01,
            rgbaColor=[1, 0.5, 0, 1.0],
            physicsClientId=self.physics_client
        )
        
        self.target_visual_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_position,
            physicsClientId=self.physics_client
        )
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        if self.physics_client is None:
            self._connect_physics()
            self._setup_underwater_scene()
            self._analyze_robot()
        
        self.current_step = 0
        
        # 初始化关节位置
        init_real = []
        for i in range(4):
            home_real = self.real_home_positions[i]
            perturbation = np.random.uniform(-0.1, 0.1)
            init_real.append(home_real + perturbation)
        init_real = np.array(init_real)
        
        init_urdf = self._real_to_urdf(init_real)
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.resetJointState(self.robot_id, joint_idx, init_urdf[i],
                            physicsClientId=self.physics_client)
        
        # 夹爪张开
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            if joint_info[1].decode('utf-8') == 'joint_5':
                p.resetJointState(self.robot_id, i, self.home_gripper_position,
                                physicsClientId=self.physics_client)
                break
        
        # ✅ 每个episode随机新的水流
        self.current_velocity = np.array([
            np.random.uniform(-0.03, 0.03),
            np.random.uniform(-0.03, 0.03),
            np.random.uniform(-0.03, 0.03)
        ], dtype=np.float32)
        self.current_velocity_actual = self.current_velocity.copy()
        
        # 生成目标位置
        self.target_position = self._sample_target_position()
        self.initial_target_position = self.target_position.copy()
        
        # ✅ 重置目标速度和历史
        self.target_velocity = np.zeros(3, dtype=np.float32)
        self.target_position_history = [self.target_position.copy()]
        
        self._create_target_visual()
        
        # 稳定仿真
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        self.previous_distance = float(np.linalg.norm(
            self._get_end_effector_position() - self.target_position
        ))
        
        observation = self._get_observation()
        
        info = {
            'target_position': self.target_position.copy(),
            'initial_distance': self.previous_distance,
            'current_real_angles': self._get_joint_positions_real().copy()
        }
        
        return observation, info
    
    def step(self, action):
        """执行一步"""
        self.current_step += 1
        action = np.array(action, dtype=np.float32)
        
        self._apply_action(action)
        
        # 应用水下力并仿真
        for _ in range(4):
            self._apply_underwater_forces()
            p.stepSimulation(physicsClientId=self.physics_client)
        
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # ✅ 更新目标位置
        self._update_target_position(dt=8./240.)
        self._update_target_visual()
        
        # ✅ 记录历史（用于下一步的速度估计）
        self.target_position_history.append(self.target_position.copy())
        if len(self.target_position_history) > self.history_length:
            self.target_position_history.pop(0)
        
        observation = self._get_observation()
        reward, success, reward_terms = self._compute_reward(action)
        
        terminated = bool(success)
        truncated = bool(self.current_step >= self.max_steps)
        
        ee_pos = self._get_end_effector_position()
        current_distance = float(np.linalg.norm(ee_pos - self.target_position))
        
        info = {
            'success': success,
            'distance': current_distance,
            'is_success': success,
            'current_velocity': self.current_velocity_actual.copy(),
            'underwater': True,
            'target_velocity_estimated': self._estimate_target_velocity().copy()  # ✅ 调试用
        }
        info.update(reward_terms)
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """关闭环境"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
            self.physics_client = None
    
    def render(self):
        """渲染"""
        pass


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("="*70)
    print("水下Alpha机械臂环境测试 - 带速度估计（无夹爪）")
    print("="*70)
    
    env = AlphaReachEnv(render_mode="human")
    
    obs, info = env.reset()
    
    print(f"\n✅ 初始化信息:")
    print(f"  观察维度: {obs.shape} (17维)")
    print(f"  动作维度: {env.action_space.shape} (4维，无夹爪)")
    print(f"  目标位置: {info['target_position']}")
    print(f"  初始距离: {info['initial_distance']:.3f}m")
    print(f"  当前真实角度: {np.degrees(info['current_real_angles']).astype(int)}°")
    
    print(f"\n✅ 观察空间分解:")
    print(f"  [0:4]   = 关节真实角度 (4维)")
    print(f"  [4:8]   = 关节速度 (4维)")
    print(f"  [8:11]  = 末端位置 (3维)")
    print(f"  [11:14] = 目标位置 (3维)")
    print(f"  [14:17] = 目标速度估计 (3维) ← 从历史位置计算")
    
    print(f"\n开始测试...")
    
    max_drift = 0.0
    
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 记录最大漂移
        if hasattr(env, 'initial_target_position'):
            drift = np.linalg.norm(env.target_position - env.initial_target_position)
            max_drift = max(max_drift, drift)
        
        if step % 50 == 0:
            # 解析观察
            joint_angles = obs[:4]
            target_vel_est = obs[14:17]
            
            print(f"\n步数 {step}:")
            print(f"  奖励={reward:.3f}, 距离={info['distance']:.3f}m, 成功={info['success']}")
            print(f"  真实角度: {np.degrees(joint_angles).astype(int)}°")
            print(f"  目标速度估计: [{target_vel_est[0]:.3f}, {target_vel_est[1]:.3f}, {target_vel_est[2]:.3f}] m/s")
            print(f"  目标漂移: {drift*100:.2f}cm / {env.max_target_drift*100:.1f}cm max")
        
        if done:
            print(f"\n{'='*70}")
            print(f"回合结束!")
            print(f"  成功: {info['success']}")
            print(f"  最终距离: {info['distance']:.3f}m")
            print(f"  最大目标漂移: {max_drift*100:.2f}cm")
            print(f"{'='*70}\n")
            
            obs, info = env.reset()
            max_drift = 0.0
            print(f"重置完成，新目标位置: {info['target_position']}")
    
    env.close()
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)
    print("\n✅ 关键特性:")
    print("  1. 动作空间: 4个关节（无夹爪）")
    print("  2. 观察空间: 17维（含目标速度估计）")
    print("  3. 速度估计: 从历史位置简单差分")
    print("  4. 目标漂移: 最大5cm，随机摆动")
    print("  5. 夹爪: 始终张开，不参与强化学习")
    print("\n✅ 部署时:")
    print("  1. 模型输出4个关节的动作")
    print("  2. 到达目标后，外部控制夹爪夹取")
    print("  3. 速度估计可用相机多帧计算")
    print("="*70)