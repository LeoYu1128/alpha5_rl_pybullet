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
    水下Alpha机械臂到达任务环境 - 带目标漂移追踪
    核心改进：
    1. 速度估计（从历史位置）
    2. 预测性奖励
    3. 速度匹配奖励
    """
    
    def __init__(self, render_mode=None, max_steps=1000, reward_type='dense'):
        super().__init__()
        
        # ===== 基础参数 =====
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.physics_client = None
        self.reward_type = reward_type
        
        # ===== 目标漂移参数（会被课程学习覆盖）=====
        self.max_target_drift = 0.15
        self.drift_damping = 0.7
        self.drift_noise_strength = 0.08
        self.target_velocity = np.zeros(3, dtype=np.float32)
        
        # ✅ Phase 1: 领域随机化
        self.enable_domain_randomization = True  # ✅ 修正属性名
        self.randomization_strength = 1.0
        self.mass_randomization_range = (0.8, 1.2)
        self.friction_randomization_range = (0.05, 0.2)
        self.damping_randomization_range = (1.5, 3.0)
        self.current_randomization_range = (-0.15, 0.15)
        self.current_randomization = {}
        
        print("✅ 领域随机化已启用")
        
        # ✅ Phase 2: 传感器噪声
        self.enable_sensor_noise = True
        self.position_noise_std = 0.003
        self.velocity_noise_std = 0.01
        self.ee_position_noise_std = 0.005
        
        print("✅ 传感器噪声已启用")
        print(f"   位置噪声: {self.position_noise_std*180/np.pi:.2f}°")
        print(f"   速度噪声: {self.velocity_noise_std:.3f} rad/s")
        
        # ✅ Phase 3: 控制延迟
        self.enable_control_delay = True
        self.control_delay_range = (0.01, 0.03)
        self.current_control_delay = 0.02
        from collections import deque
        self.action_buffer = deque(maxlen=10)
        
        print("✅ 控制延迟已启用")
        print(f"   延迟范围: {self.control_delay_range[0]*1000:.0f}-{self.control_delay_range[1]*1000:.0f} ms")
        
        # ✅ 课程学习
        self.enable_curriculum = True
        self.curriculum_stage = 0
        self.curriculum_thresholds = [10000, 30000]
        
        print("✅ 课程学习已启用")
        
        # ... 其余参数不变 ...
        
        self.urdf_total_mass = 1.52
        self.target_actual_mass = 1.36
        self.target_underwater_mass = 0.9
        buoyancy_mass = self.target_actual_mass - self.target_underwater_mass
        self.buoyancy_compensation_ratio = buoyancy_mass / self.target_actual_mass
        self.robot_mass_scale = self.target_actual_mass / self.urdf_total_mass
        
        self.drag_coefficient = 0.5
        self.buoyancy_enabled = True
        self.water_density = 1000.0
        self.gravity = -9.81
        self.base_height = 0.1
        self.workspace_radius = 0.4
        
        self.current_velocity = np.random.uniform(-0.2, 0.2, size=3)
        self.current_variation = True
        self.turbulence_strength = 0.015
        
        # 速度估计
        self.target_position_history = []
        self.history_length = 4
        self.velocity_weights = np.array([0.1, 0.2, 0.3, 0.4])
        self.dt = 1./240.
        
        # 连接物理引擎
        self._connect_physics()
        self._setup_underwater_scene()
        self._analyze_robot()
        
        # 奖励参数
        self.distance_scale = 1.0
        self.progress_scale = 2.0
        self.control_penalty = 0.01
        self.velocity_penalty = 0.02
        self.time_penalty = 0.002
        self.success_bonus = 50.0
        self.success_threshold = 0.02
        self.previous_distance = None
        self.prediction_scale = 0.8
        self.velocity_match_scale = 0.8
        self.prediction_horizon = 0.5
        
        # 动作和观察空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        obs_dim = 4 + 4 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        print("="*60)
        print("水下Alpha机械臂环境初始化完成")
        print("="*60)
    
    # ===== 以下是原有方法，保持不变 =====
    
    def _connect_physics(self):
        """连接PyBullet物理引擎"""
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
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1,
                                      physicsClientId=self.physics_client)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # 启用CCD（在 _connect_physics 中）
        p.setPhysicsEngineParameter(
            enableConeFriction=1,
            contactBreakingThreshold=0.001,
            # ✅ 启用CCD
            enableCCD=True,
            physicsClientId=self.physics_client
        )


        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1,
                                   physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1,
                                   physicsClientId=self.physics_client)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), 
                                  physicsClientId=self.physics_client)
        p.setGravity(0, 0, self.gravity * 0.1, physicsClientId=self.physics_client)
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
    
    def _setup_underwater_scene(self):
        """设置水下仿真场景"""
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.3, 0.5, 0.8, 1.0],
                           physicsClientId=self.physics_client)
        
        try:
            from pathlib import Path
            this_dir = Path(__file__).resolve().parent
            proj_root = this_dir.parent
        except Exception:
            this_dir = None
            proj_root = None

        robot_paths = [
            "alpha_robot_for_pybullet.urdf",
            "alpha_description/urdf/alpha_robot_for_pybullet.urdf",
            "../alpha_description/urdf/alpha_robot_for_pybullet.urdf",
        ]
        if proj_root is not None:
            robot_paths.extend([
                str((proj_root / "alpha_description/urdf/alpha_robot_for_pybullet.urdf").resolve()),
                str((proj_root.parent / "alpha_description/urdf/alpha_robot_for_pybullet.urdf").resolve()),
            ])
        
        self.robot_id = None
        for robot_path in robot_paths:
            if os.path.exists(robot_path):
                try:
                    self.robot_id = p.loadURDF(robot_path, basePosition=[0, 0, self.base_height],
                                              useFixedBase=True, physicsClientId=self.physics_client)
                    print(f"成功加载水下Alpha机械臂: {robot_path}")
                    break
                except Exception as e:
                    continue
        
        if self.robot_id is None:
            raise FileNotFoundError("找不到Alpha机械臂URDF文件")
        
        self._add_underwater_decorations()

    def _add_underwater_decorations(self):
        """添加水下装饰物"""
        try:
            for i in range(3):
                x = np.random.uniform(-0.3, 0.3)
                y = np.random.uniform(-0.3, 0.3)
                z = np.random.uniform(-0.3, 0.3)
                decoration_id = p.loadURDF("sphere_small.urdf", basePosition=[x, y, z],
                                          physicsClientId=self.physics_client)
                p.changeVisualShape(decoration_id, -1, rgbaColor=[0.1, 0.4, 0.2, 0.8],
                                   physicsClientId=self.physics_client)
        except:
            pass
    
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
                'name': joint_name, 'type': joint_type,
                'lower': lower_limit, 'upper': upper_limit
            }
            
            if joint_name in ['joint_1', 'joint_2', 'joint_3', 'joint_4']:
                self.main_joint_indices.append(i)
        
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
        
        # ✅ 在这里添加 - main_joint_indices 已经创建
        print("\n配置碰撞检测参数...")
        for i in self.main_joint_indices:
            p.changeDynamics(
                self.robot_id, i,
                ccdSweptSphereRadius=0.005,      # CCD球体半径
                contactStiffness=30000,          # 接触刚度（增加碰撞响应）
                contactDamping=1000,             # 接触阻尼
                collisionMargin=0.001,           # 碰撞边距（1mm）
                physicsClientId=self.physics_client
            )
        print("✅ CCD和碰撞参数已设置")
    def _setup_underwater_dynamics(self):
        """设置水下动力学"""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        self.link_masses = {}
        self.link_indices = []
        
        for i in range(-1, num_joints):
            try:
                dynamics_info = p.getDynamicsInfo(self.robot_id, i, 
                                                 physicsClientId=self.physics_client)
                urdf_mass = dynamics_info[0]
                
                if urdf_mass > 0:
                    real_mass = urdf_mass * self.robot_mass_scale
                    self.link_masses[i] = real_mass
                    self.link_indices.append(i)
                    p.changeDynamics(self.robot_id, i, mass=real_mass,
                                   physicsClientId=self.physics_client)
                
                if i >= 0:
                    p.changeDynamics(self.robot_id, i, linearDamping=2.0, angularDamping=2.0,
                                   jointDamping=0.5, physicsClientId=self.physics_client)
                else:
                    p.changeDynamics(self.robot_id, i, linearDamping=1.0, angularDamping=1.0,
                                   physicsClientId=self.physics_client)
            except:
                pass
    
    def _update_current_velocity(self):
        """更新水流速度"""
        if self.current_variation:
            time_factor = self.current_step * 0.01
            base_current = self.current_velocity.copy()
            periodic_variation = np.array([
                0.02 * np.sin(time_factor),
                0.02 * np.cos(time_factor * 1.5),
                0.02 * np.sin(time_factor * 0.5)
            ])
            turbulence = np.random.normal(0, self.turbulence_strength, 3)
            self.current_velocity_actual = base_current + periodic_variation + turbulence
        else:
            self.current_velocity_actual = self.current_velocity.copy()
    
    def _apply_underwater_forces(self):
        """应用水下力"""
        self._update_current_velocity()
        
        if self.buoyancy_enabled:
            for link_idx in self.link_indices:
                try:
                    mass = self.link_masses.get(link_idx, 0)
                    if mass > 0:
                        gravity_force = mass * abs(self.gravity)
                        buoyancy_force = gravity_force * self.buoyancy_compensation_ratio
                        p.applyExternalForce(self.robot_id, link_idx,
                                           forceObj=[0, 0, buoyancy_force],
                                           posObj=[0, 0, 0], flags=p.LINK_FRAME,
                                           physicsClientId=self.physics_client)
                except:
                    pass
        
        ee_state = p.getLinkState(self.robot_id, self.tcp_index, computeLinkVelocity=1,
                                  physicsClientId=self.physics_client)
        ee_velocity = np.array(ee_state[6])
        relative_velocity = ee_velocity - self.current_velocity_actual
        
        drag_force = -0.5 * self.water_density * self.drag_coefficient * 0.01 * \
                     relative_velocity * np.linalg.norm(relative_velocity)
        max_force = 5.0
        drag_force = np.clip(drag_force, -max_force, max_force)
        
        p.applyExternalForce(self.robot_id, self.tcp_index, forceObj=drag_force,
                           posObj=[0, 0, 0], flags=p.LINK_FRAME,
                           physicsClientId=self.physics_client)
        
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx, 
                                         physicsClientId=self.physics_client)
            joint_velocity = joint_state[1]
            damping_torque = -0.1 * joint_velocity
            p.setJointMotorControl2(self.robot_id, joint_idx, p.TORQUE_CONTROL,
                                  force=damping_torque, physicsClientId=self.physics_client)
    
    def _update_target_position(self, dt=1./240.):
        """更新目标位置"""
        if not hasattr(self, 'target_position'):
            return
        
        displacement = self.target_position - self.initial_target_position
        distance_from_initial = np.linalg.norm(displacement)
        
        if distance_from_initial > 0.001:
            restore_strength = min(distance_from_initial / self.max_target_drift, 1.0) ** 2
            restore_force = -displacement * restore_strength * 0.3
        else:
            restore_force = np.zeros(3)
        
        random_current = np.array([
            np.random.uniform(-self.drift_noise_strength, self.drift_noise_strength),
            np.random.uniform(-self.drift_noise_strength, self.drift_noise_strength),
            np.random.uniform(-self.drift_noise_strength * 0.5, self.drift_noise_strength * 0.5)
        ])
        
        time_factor = self.current_step * 0.01
        wave_force = np.array([
            0.02 * np.sin(time_factor * 0.2),
            0.02 * np.cos(time_factor * 0.35),
            0.02 * np.sin(time_factor * 0.2)
        ])
        
        total_force = restore_force + random_current + wave_force
        self.target_velocity += total_force
        self.target_velocity *= self.drift_damping
        
        max_velocity = 0.02
        velocity_magnitude = np.linalg.norm(self.target_velocity)
        if velocity_magnitude > max_velocity:
            self.target_velocity *= (max_velocity / velocity_magnitude)
        
        self.target_position += self.target_velocity * dt
        
        displacement = self.target_position - self.initial_target_position
        distance_from_initial = np.linalg.norm(displacement)
        
        if distance_from_initial > self.max_target_drift:
            direction = displacement / distance_from_initial
            self.target_position = self.initial_target_position + direction * self.max_target_drift
            self.target_velocity = -self.target_velocity * 0.3
        
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

    def _sample_target_position(self):
        """采样目标位置"""
        safe_radius_min = 0.12
        safe_radius_max = 0.4
        z_min = 0.15
        z_max = 0.35

        r = np.random.uniform(safe_radius_min, safe_radius_max)
        theta = np.random.uniform(-np.pi/2, np.pi/2)
        z = np.random.uniform(z_min, z_max)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return np.array([x, y, z], dtype=np.float32)

    def _get_joint_positions(self):
        """获取关节位置 - 带噪声版"""
        positions = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx,
                                        physicsClientId=self.physics_client)
            true_position = joint_state[0]
            
            # ✅ Phase 2: 添加位置噪声
            if self.enable_sensor_noise:
                noise = np.random.normal(0, self.position_noise_std)
                noisy_position = true_position + noise
                
                # 限制在关节范围内
                joint_info = self.joint_info[joint_idx]
                noisy_position = np.clip(noisy_position, 
                                        joint_info['lower'], 
                                        joint_info['upper'])
            else:
                noisy_position = true_position
            
            positions.append(noisy_position)
        
        return np.array(positions, dtype=np.float32)
    
    def _get_joint_velocities(self):
        """获取关节速度 - 带噪声版"""
        velocities = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx,
                                        physicsClientId=self.physics_client)
            true_velocity = joint_state[1]
            
            # ✅ Phase 2: 添加速度噪声
            if self.enable_sensor_noise:
                noise = np.random.normal(0, self.velocity_noise_std)
                noisy_velocity = true_velocity + noise
            else:
                noisy_velocity = true_velocity
            
            velocities.append(noisy_velocity)
        
        return np.array(velocities, dtype=np.float32)
    
    def _get_end_effector_position(self):
        """获取末端位置 - 带噪声版"""
        link_state = p.getLinkState(self.robot_id, self.tcp_index,
                                    physicsClientId=self.physics_client)
        true_position = np.array(link_state[0], dtype=np.float32)
        
        # ✅ Phase 2: 添加末端位置噪声
        if self.enable_sensor_noise:
            noise = np.random.normal(0, self.ee_position_noise_std, size=3)
            noisy_position = true_position + noise
        else:
            noisy_position = true_position
        
        return noisy_position
    
    # ===== ✅ 新增：获取末端速度 =====
    def _get_end_effector_velocity(self):
        """获取末端执行器速度"""
        link_state = p.getLinkState(self.robot_id, self.tcp_index, 
                                    computeLinkVelocity=1,
                                    physicsClientId=self.physics_client)
        return np.array(link_state[6], dtype=np.float32)
    
    # ===== ✅ 新增：目标速度估计（核心方法）=====
    def _estimate_target_velocity(self):
        """
        从历史位置估计目标速度（加权平均法）
        使用4帧历史，近期权重更高
        """
        if len(self.target_position_history) < 2:
            return np.zeros(3, dtype=np.float32)
        
        velocities = []
        available_frames = min(self.history_length, len(self.target_position_history) - 1)
        
        # 计算每一帧的速度
        for i in range(available_frames):
            curr_pos = self.target_position_history[-(i+1)]
            prev_pos = self.target_position_history[-(i+2)]
            v = (curr_pos - prev_pos) / self.dt
            velocities.append(v)
        
        # 加权平均（近期权重更高）
        weights = self.velocity_weights[:len(velocities)]
        weights = weights / weights.sum()  # 归一化
        
        estimated_velocity = np.average(velocities, axis=0, weights=weights)
        
        # 限制最大速度（防止异常值）
        max_velocity = 0.2
        velocity_magnitude = np.linalg.norm(estimated_velocity)
        if velocity_magnitude > max_velocity:
            estimated_velocity *= (max_velocity / velocity_magnitude)
        
        return estimated_velocity.astype(np.float32)
    
    def _apply_action(self, action):
        """应用动作 - 带延迟版"""
        
        # ✅ Phase 3: 控制延迟
        if self.enable_control_delay:
            # 将当前动作加入缓冲区
            self.action_buffer.append(action.copy())
            
            # 计算延迟步数
            delay_steps = int(self.current_control_delay / (1./240.))
            
            # 如果缓冲区长度足够，使用延迟后的动作
            if len(self.action_buffer) > delay_steps:
                delayed_action = self.action_buffer[0]
                
                # 调试信息（每100步打印一次）
                if self.current_step % 100 == 0:
                    delay_ms = self.current_control_delay * 1000
                    print(f"  应用延迟动作: {delay_ms:.1f}ms 前的动作")
            else:
                # 缓冲区不够长，使用当前动作（episode开始时）
                delayed_action = action
        else:
            delayed_action = action
        
        # ===== 以下是原有的 _apply_action 逻辑 =====
        current_positions = self._get_joint_positions()
        current_distance = np.linalg.norm(self._get_end_effector_position() - self.target_position)
        
        # 自适应缩放
        if current_distance > 0.15:
            scale = 0.08
        elif current_distance > 0.07:
            scale = 0.06
        else:
            scale = 0.03
        
        scaled_action = delayed_action * scale  # ✅ 使用延迟后的动作
        target_positions = current_positions + scaled_action
        
        # 应用关节限制
        for i, joint_idx in enumerate(self.main_joint_indices):
            joint = self.joint_info[joint_idx]
            target_positions[i] = np.clip(target_positions[i], joint['lower'], joint['upper'])
        
        # 应用控制
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.setJointMotorControl2(
                self.robot_id, 
                joint_idx, 
                p.POSITION_CONTROL,
                targetPosition=target_positions[i], 
                maxVelocity=0.6,  # 根据前面讨论调整
                force=8.0,        # 根据前面讨论调整
                physicsClientId=self.physics_client
            )
    def _check_self_collision(self):
        """检查自碰撞"""
        contact_points = p.getContactPoints(
            bodyA=self.robot_id, 
            bodyB=self.robot_id,
            physicsClientId=self.physics_client
        )
        
        collision_count = 0
        for contact in contact_points:
            link_a = contact[3]
            link_b = contact[4]
            
            # 只计算非相邻关节的碰撞
            if abs(link_a - link_b) > 1:
                collision_count += 1
        
        return collision_count

    # ===== ✅ 修改：奖励函数（添加预测性奖励）=====
    def _compute_reward(self, action):
        """
        组合奖励函数
        新增：预测性奖励 + 速度匹配奖励
        """
        ee_pos = self._get_end_effector_position()
        ee_vel = self._get_end_effector_velocity()
        target_pos = self.target_position
        target_vel = self._estimate_target_velocity()  # ✅ 使用速度估计
        
        # 1. 当前距离奖励
        current_distance = float(np.linalg.norm(ee_pos - target_pos))
        distance_reward = -self.distance_scale * current_distance
        
        # ✅ 2. 预测距离奖励（核心！）
        predicted_target_pos = target_pos + target_vel * self.prediction_horizon
        predicted_distance = float(np.linalg.norm(ee_pos - predicted_target_pos))
        prediction_reward = -self.prediction_scale * predicted_distance
        
        # ✅ 3. 速度匹配奖励（核心！）
        velocity_error = float(np.linalg.norm(ee_vel - target_vel))
        velocity_match_reward = 1.0 -self.velocity_match_scale * velocity_error
        
        # 4. 进步奖励
        # prev_distance = self.previous_distance if self.previous_distance else current_distance
        # distance_delta = prev_distance - current_distance
        # progress_reward_distance = self.progress_scale * distance_delta
        direction_to_target = (target_pos - ee_pos) / (current_distance + 1e-6)
        approach_velocity = float(np.dot(ee_vel, direction_to_target))
        # ✅ 正速度（接近）→ 正奖励，负速度（远离）→ 负奖励
        progress_reward_vel = self.progress_scale * approach_velocity
        
        # 5. 控制代价
        control_cost = self.control_penalty * float(np.linalg.norm(action) ** 2)
        velocity_cost = self.velocity_penalty * float(np.linalg.norm(self._get_joint_velocities()))
        time_cost = self.time_penalty
        # ✅ 6. 碰撞惩罚（新增）
        collision_count = self._check_self_collision()
        collision_penalty = 10.0 * collision_count  # 每次碰撞 -10 奖励（巨大惩罚）
        # 6. 成功判断
        success = current_distance < self.success_threshold
        success_bonus = self.success_bonus if success else 0.0
        
        # 总奖励
        total_reward = (
            distance_reward + 
            prediction_reward +      # ✅ 新增
            velocity_match_reward +  # ✅ 新增
            progress_reward_vel - 
            control_cost - 
            velocity_cost - 
            collision_penalty -
            time_cost + 
            success_bonus
        )
        
        self.previous_distance = current_distance
        
        reward_terms = {
            'reward_distance': distance_reward,
            'reward_prediction': prediction_reward,      # ✅ 新增
            'reward_velocity_match': velocity_match_reward,  # ✅ 新增
            'reward_progress_vel': progress_reward_vel,
            'reward_control': -control_cost,
            'reward_velocity': -velocity_cost,
            'reward_collision': -collision_penalty,  # ✅ 记录碰撞
            'reward_time': -time_cost,
            'reward_success': success_bonus
        }
         # ✅ 如果有碰撞，打印警告
        if collision_count > 0 and self.current_step % 10 == 0:
            print(f"⚠️ Step {self.current_step}: 检测到 {collision_count} 次碰撞！")
        
        return float(total_reward), bool(success), reward_terms
    
    # ===== ✅ 修改：观察空间（添加目标速度）=====
    def _get_observation(self):
        """
        获取观察
        新增：目标速度估计（3维）
        """
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        ee_position = self._get_end_effector_position()
        
        # ✅ 估计目标速度
        target_velocity = self._estimate_target_velocity()
        
        observation = np.concatenate([
            joint_positions,       # 4维
            joint_velocities,      # 4维  
            ee_position,           # 3维
            self.target_position,  # 3维
            target_velocity        # 3维 ← ✅ 新增
        ]).astype(np.float32)
        
        return observation
    
    def _create_target_visual(self):
        """创建目标视觉"""
        if hasattr(self, 'target_visual_id'):
            try:
                p.removeBody(self.target_visual_id, physicsClientId=self.physics_client)
            except:
                pass
        
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.01,
                                          rgbaColor=[1, 0.5, 0, 1.0],
                                          physicsClientId=self.physics_client)
        
        self.target_visual_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape,
                                                  basePosition=self.target_position,
                                                  physicsClientId=self.physics_client)
    
    def _update_target_visual(self):
        """更新目标视觉"""
        if hasattr(self, 'target_visual_id') and self.target_visual_id is not None:
            try:
                p.resetBasePositionAndOrientation(self.target_visual_id, self.target_position,
                                                 [0, 0, 0, 1], physicsClientId=self.physics_client)
            except:
                pass

    def _randomize_dynamics(self):
        """
        领域随机化：每个episode开始时随机化物理参数
        模拟真实世界的不确定性和变化
        """
        if not self.enable_domain_randomization:
            return
        
        print(f"\n{'='*60}")
        print(f"Episode {getattr(self, 'episode_count', 0)}: 应用领域随机化")
        print(f"{'='*60}")
        
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        # ===== 1. 随机化连杆质量 =====
        for link_idx in self.link_indices:
            try:
                # 获取基础质量
                base_mass = self.link_masses.get(link_idx, 0.1)
                
                # 随机化
                mass_scale = np.random.uniform(*self.mass_randomization_range)
                random_mass = base_mass * mass_scale
                
                # 应用
                p.changeDynamics(
                    self.robot_id, link_idx,
                    mass=random_mass,
                    physicsClientId=self.physics_client
                )
                
                # 记录
                if link_idx == self.link_indices[0]:  # 只打印第一个
                    print(f"  质量随机化: {base_mass:.3f} → {random_mass:.3f} kg (x{mass_scale:.2f})")
            
            except Exception as e:
                print(f"  警告: Link {link_idx} 质量随机化失败: {e}")
        
        # ===== 2. 随机化摩擦系数 =====
        friction_coef = np.random.uniform(*self.friction_randomization_range)
        
        for link_idx in range(-1, num_joints):
            try:
                p.changeDynamics(
                    self.robot_id, link_idx,
                    lateralFriction=friction_coef,
                    physicsClientId=self.physics_client
                )
            except:
                pass
        
        print(f"  摩擦随机化: {friction_coef:.3f}")
        
        # ===== 3. 随机化阻尼 =====
        linear_damping = np.random.uniform(*self.damping_randomization_range)
        angular_damping = np.random.uniform(*self.damping_randomization_range)
        
        for i in range(num_joints):
            try:
                p.changeDynamics(
                    self.robot_id, i,
                    linearDamping=linear_damping,
                    angularDamping=angular_damping,
                    physicsClientId=self.physics_client
                )
            except:
                pass
        
        print(f"  阻尼随机化: linear={linear_damping:.2f}, angular={angular_damping:.2f}")
        
        # ===== 4. 随机化水流速度 =====
        self.current_velocity = np.random.uniform(
            self.current_randomization_range[0],
            self.current_randomization_range[1],
            size=3
        )
        print(f"  水流随机化: [{self.current_velocity[0]:.3f}, {self.current_velocity[1]:.3f}, {self.current_velocity[2]:.3f}] m/s")
        
        # ===== 5. 随机化重力（轻微）=====
        gravity_scale = np.random.uniform(0.95, 1.05)  # ±5%
        randomized_gravity = self.gravity * gravity_scale * 0.1  # 保持水下重力
        p.setGravity(0, 0, randomized_gravity, physicsClientId=self.physics_client)
        print(f"  重力随机化: {randomized_gravity:.3f} m/s² (scale={gravity_scale:.3f})")
        # ===== 6. 随机化控制延迟 =====
        if self.enable_control_delay:
            self.current_control_delay = np.random.uniform(*self.control_delay_range)
            delay_steps = int(self.current_control_delay / (1./240.))
            print(f"  控制延迟: {self.current_control_delay*1000:.1f} ms ({delay_steps} 物理步)")
        # 保存当前随机化参数（用于分析）
        self.current_randomization = {
            'mass_scale': mass_scale,
            'friction': friction_coef,
            'damping': (linear_damping, angular_damping),
            'current_velocity': self.current_velocity.copy(),
            'gravity_scale': gravity_scale
        }
        
        print(f"{'='*60}\n")

    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        if self.physics_client is None:
            self._connect_physics()
            self._setup_underwater_scene()
            self._analyze_robot()
        
        self.current_step = 0
        
        # ===== 1. 随机初始化关节位置 =====
        init_positions = []
        for joint_idx in self.main_joint_indices:
            joint = self.joint_info[joint_idx]
            range_center = (joint['lower'] + joint['upper']) / 2
            range_width = (joint['upper'] - joint['lower']) * 0.8
            init_pos = np.random.uniform(range_center - range_width, range_center + range_width)
            init_positions.append(init_pos)
        
        # ===== 2. 重置主关节 =====
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.resetJointState(self.robot_id, joint_idx, init_positions[i],
                            physicsClientId=self.physics_client)
        
        # ===== 3. 重置夹爪到home位置 =====
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            
            if joint_name == 'joint_5':
                p.resetJointState(self.robot_id, i, 0.0014, physicsClientId=self.physics_client)
                # 只在第一次打印
                if self.current_step == 0 and not hasattr(self, 'episode_count'):
                    print(f"  Reset {joint_name} to 0.0014 (home)")
            elif 'jaw' in joint_name.lower() or 'gripper' in joint_name.lower():
                home_angle = 0.0014 * 41.67
                p.resetJointState(self.robot_id, i, home_angle, physicsClientId=self.physics_client)
                if self.current_step == 0 and not hasattr(self, 'episode_count'):
                    print(f"  Reset {joint_name} to {home_angle:.4f}")
        
        # ===== 4. Episode计数 =====
        if not hasattr(self, 'episode_count'):
            self.episode_count = 0
        self.episode_count += 1
        
        # ===== 5. 清空动作缓冲区 =====
        if hasattr(self, 'action_buffer'):
            self.action_buffer.clear()
        
        # ✅ 6. 应用领域随机化（必须在if外面！）
        self._randomize_dynamics()
        
        # ===== 7. 重置目标位置 =====
        self.target_position = self._sample_target_position()
        self.initial_target_position = self.target_position.copy()
        self.target_velocity = np.zeros(3, dtype=np.float32)
        
        # ===== 8. 重置目标漂移参数（Phase 4可选）=====
        if self.enable_domain_randomization and hasattr(self, 'randomization_strength'):
            # 随机化目标漂移参数
            self.max_target_drift = np.random.uniform(0.10, 0.20)
            self.drift_damping = np.random.uniform(0.6, 0.8)
            self.drift_noise_strength = np.random.uniform(0.05, 0.12)
        
        # ===== 9. 重置历史缓冲 =====
        self.target_position_history = [self.target_position.copy()]
        
        # ===== 10. 重置水流 =====
        self.current_velocity_actual = self.current_velocity.copy()
        
        # ===== 11. 创建目标视觉 =====
        self._create_target_visual()
        
        # ===== 12. 稳定仿真 =====
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # ===== 13. 计算初始距离 =====
        self.previous_distance = float(np.linalg.norm(
            self._get_end_effector_position() - self.target_position
        ))
        
        # ===== 14. 获取观察 =====
        observation = self._get_observation()
        
        # ===== 15. 构建info =====
        info = {
            'target_position': self.target_position.copy(),
            'initial_distance': self.previous_distance,
            'episode_count': self.episode_count,  # 添加episode计数
            'randomization_strength': getattr(self, 'randomization_strength', 1.0)  # 添加随机化强度
        }
        
        return observation, info
    
    def step(self, action):
        """执行一步"""
        self.current_step += 1
        action = np.array(action, dtype=np.float32)
        
        self._apply_action(action)
        self._apply_underwater_forces()
        
        for _ in range(16):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        self._update_target_position(dt=4./240.)
        self._update_target_visual()
        
        # ✅ 更新历史缓冲（用于下一步速度估计）
        self.target_position_history.append(self.target_position.copy())
        if len(self.target_position_history) > self.history_length + 1:
            self.target_position_history.pop(0)
        
        observation = self._get_observation()
        reward, success, reward_terms = self._compute_reward(action)
        
        terminated = bool(success)
        truncated = bool(self.current_step >= self.max_steps)
        
        ee_pos = self._get_end_effector_position()
        current_distance = float(np.linalg.norm(ee_pos - self.target_position))
        
        if hasattr(self, 'initial_target_position'):
            current_drift = float(np.linalg.norm(self.target_position - self.initial_target_position))
        else:
            current_drift = 0.0
        
        info = {
            'success': success,
            'distance': current_distance,
            'is_success': success,
            'current_velocity': self.current_velocity_actual.copy(),
            'underwater': True,
            'target_drift': current_drift,
            'target_velocity': self.target_velocity.copy(),
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

# ===== 测试代码 =====
if __name__ == "__main__":
    print("\n" + "="*70)
    print("水下Alpha机械臂环境测试 - 带目标漂移追踪")
    print("="*70 + "\n")
    
    env = AlphaReachEnv(render_mode="human")
    obs, info = env.reset()
    
    print(f"✅ 观察维度: {obs.shape} (17维)")
    print(f"   [0:4]   = 关节位置")
    print(f"   [4:8]   = 关节速度")
    print(f"   [8:11]  = 末端位置")
    print(f"   [11:14] = 目标位置")
    print(f"   [14:17] = 目标速度估计 ← 新增\n")
    
    print(f"初始目标位置: {info['target_position']}")
    print(f"最大漂移限制: {env.max_target_drift * 100:.1f}cm\n")
    
    max_drift_observed = 0.0
    
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if 'target_drift' in info:
            max_drift_observed = max(max_drift_observed, info['target_drift'])
        
        if step % 50 == 0:
            drift_cm = info.get('target_drift', 0) * 100
            target_vel_est = info.get('target_velocity_estimated', np.zeros(3))
            
            print(f"步数 {step}:")
            print(f"  距离: {info['distance']:.3f}m")
            print(f"  目标漂移: {drift_cm:.2f}cm")
            print(f"  估计速度: [{target_vel_est[0]:.3f}, {target_vel_est[1]:.3f}, {target_vel_est[2]:.3f}] m/s")
            
            # 打印奖励分解
            if step == 50:
                print(f"  奖励分解:")
                for key, value in info.items():
                    if key.startswith('reward_'):
                        print(f"    {key}: {value:.3f}")
        
        if done:
            print(f"\n{'='*70}")
            print(f"回合结束:")
            print(f"  最大漂移: {max_drift_observed * 100:.2f}cm / {env.max_target_drift * 100:.1f}cm")
            print(f"  成功: {info['success']}")
            print(f"  最终距离: {info['distance']:.3f}m")
            print(f"{'='*70}\n")
            
            obs, info = env.reset()
            max_drift_observed = 0.0
    
    env.close()
    print("\n测试完成!")