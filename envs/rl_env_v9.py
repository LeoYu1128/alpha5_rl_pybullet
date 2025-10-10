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
    水下Alpha机械臂到达任务环境 - 带明显目标漂移追踪
    核心改进：
    1. 增强目标漂移（最大20cm，明显可见）
    2. 速度估计（从历史位置）
    3. 预测性奖励
    4. 碰撞检测和惩罚
    5. 课程学习
    """
    
    def __init__(self, render_mode=None, max_steps=1000, reward_type='dense'):
        super().__init__()
        
        # ===== 基础参数 =====
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.physics_client = None
        self.reward_type = reward_type
        
        # ===== ✅ 目标漂移参数（显著增强！）=====
        self.max_target_drift = 0.1  # 20cm漂移（原来0.04只有4cm）
        self.drift_damping = 0.85     # 降低阻尼，让漂移更持久
        self.drift_noise_strength = 0.15  # 增强随机扰动
        self.target_velocity = np.zeros(3, dtype=np.float32)
        
        # ✅ Phase 1: 领域随机化
        self.enable_domain_randomization = True
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
        self.curriculum_thresholds = [200, 800]
        
        print("✅ 课程学习已启用")
        print("   阶段0 (0-10k ep): 小漂移5cm + 弱随机化")
        print("   阶段1 (10k-30k ep): 中漂移12cm + 中随机化")
        print("   阶段2 (30k+ ep): 大漂移20cm + 强随机化")
        
        # ===== 物理参数 =====
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
        
        # ===== 速度估计 =====
        self.target_position_history = []
        self.history_length = 4
        self.velocity_weights = np.array([0.1, 0.2, 0.3, 0.4])
        self.dt = 1./240.
        
        # 连接物理引擎
        self._connect_physics()
        self._setup_underwater_scene()
        self._analyze_robot()
        
        # ===== 奖励参数 =====
        self.distance_scale = 1.0
        self.progress_scale = 2.0
        self.control_penalty = 0.01
        self.velocity_penalty = 0.02
        self.time_penalty = 0.002
        self.success_bonus = 50.0
        self.success_threshold = 0.03  # 放宽成功阈值（因为目标在动）
        self.previous_distance = None
        self.prediction_scale = 0.8
        self.velocity_match_scale = 0.8
        self.prediction_horizon = 0.5
        
        # 动作和观察空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        obs_dim = 4 + 4 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        print("="*60)
        print("水下Alpha机械臂环境初始化完成 - 强漂移版")
        print(f"最大目标漂移: {self.max_target_drift*100:.0f}cm (明显可见)")
        print("="*60)
    
    def _connect_physics(self):
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
        p.setGravity(0, 0, self.gravity * 0.1, physicsClientId=self.physics_client)
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
    
    def _setup_underwater_scene(self):
        """设置水下仿真场景"""
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.3, 0.5, 0.8, 1.0],
                           physicsClientId=self.physics_client)
        
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
                        physicsClientId=self.physics_client
                    )
                    print(f"成功加载: {robot_path}")
                    break
                except:
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
        
        # # 配置CCD防穿模
        # print("\n配置防穿模参数...")
        # for i in self.main_joint_indices:
        #     p.changeDynamics(
        #         self.robot_id, i,
        #         ccdSweptSphereRadius=0.005,
        #         contactStiffness=30000,
        #         contactDamping=1000,
        #         collisionMargin=0.001,
        #         physicsClientId=self.physics_client
        #     )
        print("✅ 防穿模配置完成")
    
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
                    p.changeDynamics(self.robot_id, i, 
                                   linearDamping=2.0, 
                                   angularDamping=2.0,
                                   jointDamping=0.5, 
                                   physicsClientId=self.physics_client)
                else:
                    p.changeDynamics(self.robot_id, i, 
                                   linearDamping=1.0, 
                                   angularDamping=1.0,
                                   physicsClientId=self.physics_client)
            except:
                pass
    
    def _update_current_velocity(self):
        """更新水流速度"""
        if self.current_variation:
            time_factor = self.current_step * 0.01
            base_current = self.current_velocity.copy()
            periodic_variation = np.array([
                0.03 * np.sin(time_factor),
                0.03 * np.cos(time_factor * 1.5),
                0.02 * np.sin(time_factor * 0.5)
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
        
        # 末端阻力
        try:
            ee_state = p.getLinkState(
                self.robot_id, self.tcp_index, 
                computeLinkVelocity=1,
                physicsClientId=self.physics_client
            )
            ee_velocity = np.array(ee_state[6])
            relative_velocity = ee_velocity - self.current_velocity_actual
            
            drag_force = -0.5 * self.water_density * self.drag_coefficient * 0.01 * \
                         relative_velocity * np.linalg.norm(relative_velocity)
            max_force = 5.0
            drag_force = np.clip(drag_force, -max_force, max_force)
            
            p.applyExternalForce(
                self.robot_id, self.tcp_index, 
                forceObj=drag_force,
                posObj=[0, 0, 0], 
                flags=p.LINK_FRAME,
                physicsClientId=self.physics_client
            )
        except:
            pass
        
        # 关节阻尼
        for joint_idx in self.main_joint_indices:
            try:
                joint_state = p.getJointState(
                    self.robot_id, joint_idx, 
                    physicsClientId=self.physics_client
                )
                joint_velocity = joint_state[1]
                damping_torque = -0.1 * joint_velocity
                p.setJointMotorControl2(
                    self.robot_id, joint_idx, 
                    p.TORQUE_CONTROL,
                    force=damping_torque, 
                    physicsClientId=self.physics_client
                )
            except:
                pass
    
    def _update_target_position(self, dt=1./240.):
        """✅ 更新目标位置 - 增强漂移！"""
        if not hasattr(self, 'target_position'):
            return
        
        # 1. 恢复力（拉回初始位置）
        displacement = self.target_position - self.initial_target_position
        distance_from_initial = np.linalg.norm(displacement)
        
        if distance_from_initial > 0.001:
            restore_strength = min(distance_from_initial / self.max_target_drift, 1.0) ** 2
            restore_force = -displacement * restore_strength * 0.2  # 减弱恢复力
        else:
            restore_force = np.zeros(3)
        
        # 2. 随机扰动（增强）
        random_current = np.array([
            np.random.uniform(-self.drift_noise_strength, self.drift_noise_strength),
            np.random.uniform(-self.drift_noise_strength, self.drift_noise_strength),
            np.random.uniform(-self.drift_noise_strength * 0.5, self.drift_noise_strength * 0.5)
        ])
        
        # 3. 波浪力（增强）
        time_factor = self.current_step * 0.01
        wave_force = np.array([
            0.05 * np.sin(time_factor * 0.3),      # 增大振幅
            0.05 * np.cos(time_factor * 0.4),      # 增大振幅
            0.03 * np.sin(time_factor * 0.25)      # 增大振幅
        ])
        
        # 4. 水流推动力（新增）
        water_push = self.current_velocity_actual * 0.3
        
        # 总力
        total_force = restore_force + random_current + wave_force + water_push
        self.target_velocity += total_force
        self.target_velocity *= self.drift_damping
        
        # 限制最大速度
        max_velocity = 0.05  # 增大到5cm/s（原来2cm/s）
        velocity_magnitude = np.linalg.norm(self.target_velocity)
        if velocity_magnitude > max_velocity:
            self.target_velocity *= (max_velocity / velocity_magnitude)
        
        # 更新位置
        self.target_position += self.target_velocity * dt
        
        # 限制漂移范围
        displacement = self.target_position - self.initial_target_position
        distance_from_initial = np.linalg.norm(displacement)
        
        if distance_from_initial > self.max_target_drift:
            direction = displacement / distance_from_initial
            self.target_position = self.initial_target_position + direction * self.max_target_drift
            self.target_velocity = -self.target_velocity * 0.5
        
        # 限制在工作空间内
        x, y, z = self.target_position
        distance_from_base = np.sqrt(x**2 + y**2 + (z - self.base_height)**2)
        
        if distance_from_base > 0.35:
            scale = 0.35 / distance_from_base
            self.target_position = np.array([x * scale, y * scale, z], dtype=np.float32)
            self.target_velocity *= 0.5
        
        # 高度限制
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
        safe_radius_max = 0.35
        z_min = 0.15
        z_max = 0.35
        
        r = np.random.uniform(safe_radius_min, safe_radius_max)
        theta = np.random.uniform(-np.pi/2, np.pi/2)
        z = np.random.uniform(z_min, z_max)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return np.array([x, y, z], dtype=np.float32)
    
    def _get_joint_positions(self):
        """获取关节位置"""
        positions = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(
                self.robot_id, joint_idx,
                physicsClientId=self.physics_client
            )
            true_position = joint_state[0]
            
            if self.enable_sensor_noise:
                noise = np.random.normal(0, self.position_noise_std)
                noisy_position = true_position + noise
                joint_info = self.joint_info[joint_idx]
                noisy_position = np.clip(
                    noisy_position, 
                    joint_info['lower'], 
                    joint_info['upper']
                )
            else:
                noisy_position = true_position
            
            positions.append(noisy_position)
        
        return np.array(positions, dtype=np.float32)
    
    def _get_joint_velocities(self):
        """获取关节速度"""
        velocities = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(
                self.robot_id, joint_idx,
                physicsClientId=self.physics_client
            )
            true_velocity = joint_state[1]
            
            if self.enable_sensor_noise:
                noise = np.random.normal(0, self.velocity_noise_std)
                noisy_velocity = true_velocity + noise
            else:
                noisy_velocity = true_velocity
            
            velocities.append(noisy_velocity)
        
        return np.array(velocities, dtype=np.float32)
    
    def _get_end_effector_position(self):
        """获取末端位置"""
        link_state = p.getLinkState(
            self.robot_id, self.tcp_index,
            physicsClientId=self.physics_client
        )
        true_position = np.array(link_state[0], dtype=np.float32)
        
        if self.enable_sensor_noise:
            noise = np.random.normal(0, self.ee_position_noise_std, size=3)
            noisy_position = true_position + noise
        else:
            noisy_position = true_position
        
        return noisy_position
    
    def _get_end_effector_velocity(self):
        """获取末端速度"""
        link_state = p.getLinkState(
            self.robot_id, self.tcp_index, 
            computeLinkVelocity=1,
            physicsClientId=self.physics_client
        )
        return np.array(link_state[6], dtype=np.float32)
    
    def _estimate_target_velocity(self):
        """估计目标速度"""
        if len(self.target_position_history) < 2:
            return np.zeros(3, dtype=np.float32)
        
        velocities = []
        available_frames = min(self.history_length, len(self.target_position_history) - 1)
        
        for i in range(available_frames):
            curr_pos = self.target_position_history[-(i+1)]
            prev_pos = self.target_position_history[-(i+2)]
            v = (curr_pos - prev_pos) / self.dt
            velocities.append(v)
        
        weights = self.velocity_weights[:len(velocities)]
        weights = weights / weights.sum()
        
        estimated_velocity = np.average(velocities, axis=0, weights=weights)
        
        max_velocity = 0.2
        velocity_magnitude = np.linalg.norm(estimated_velocity)
        if velocity_magnitude > max_velocity:
            estimated_velocity *= (max_velocity / velocity_magnitude)
        
        return estimated_velocity.astype(np.float32)
    
    def _apply_action(self, action):
        """应用动作"""
        if self.enable_control_delay:
            self.action_buffer.append(action.copy())
            delay_steps = int(self.current_control_delay / (1./240.))
            
            if len(self.action_buffer) > delay_steps:
                delayed_action = self.action_buffer[0]
            else:
                delayed_action = action
        else:
            delayed_action = action
        
        current_positions = self._get_joint_positions()
        current_distance = np.linalg.norm(
            self._get_end_effector_position() - self.target_position
        )
        
        if current_distance > 0.15:
            scale = 0.08
        elif current_distance > 0.07:
            scale = 0.06
        else:
            scale = 0.03
        
        scaled_action = delayed_action * scale
        target_positions = current_positions + scaled_action
        
        for i, joint_idx in enumerate(self.main_joint_indices):
            joint = self.joint_info[joint_idx]
            target_positions[i] = np.clip(
                target_positions[i], 
                joint['lower'], 
                joint['upper']
            )
        
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.setJointMotorControl2(
                self.robot_id, 
                joint_idx, 
                p.POSITION_CONTROL,
                targetPosition=target_positions[i], 
                maxVelocity=0.6,
                force=8.0,
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
            
            if abs(link_a - link_b) > 1:
                collision_count += 1
        
        return collision_count
    
    def _compute_reward(self, action):
        """计算奖励"""
        ee_pos = self._get_end_effector_position()
        ee_vel = self._get_end_effector_velocity()
        target_pos = self.target_position
        target_vel = self._estimate_target_velocity()
        
        # 1. 距离奖励
        current_distance = float(np.linalg.norm(ee_pos - target_pos))
        distance_reward = -self.distance_scale * current_distance
        
        # 2. 预测奖励
        predicted_target_pos = target_pos + target_vel * self.prediction_horizon
        predicted_distance = float(np.linalg.norm(ee_pos - predicted_target_pos))
        prediction_reward = -self.prediction_scale * predicted_distance
        
        # 3. 速度匹配
        velocity_error = float(np.linalg.norm(ee_vel - target_vel))
        velocity_match_reward = 1.0 - self.velocity_match_scale * velocity_error
        
        # 4. 接近速度
        direction_to_target = (target_pos - ee_pos) / (current_distance + 1e-6)
        approach_velocity = float(np.dot(ee_vel, direction_to_target))
        progress_reward_vel = self.progress_scale * approach_velocity
        
        # 5. 控制代价
        control_cost = self.control_penalty * float(np.linalg.norm(action) ** 2)
        velocity_cost = self.velocity_penalty * float(np.linalg.norm(self._get_joint_velocities()))
        time_cost = self.time_penalty
        
        # 6. 碰撞惩罚
        collision_count = self._check_self_collision()
        collision_penalty = 10.0 * collision_count
        
        # 7. 成功判断
        success = current_distance < self.success_threshold
        success_bonus = self.success_bonus if success else 0.0
        
        total_reward = (
            distance_reward + 
            prediction_reward +
            velocity_match_reward +
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
            'reward_prediction': prediction_reward,
            'reward_velocity_match': velocity_match_reward,
            'reward_progress_vel': progress_reward_vel,
            'reward_control': -control_cost,
            'reward_velocity': -velocity_cost,
            'reward_collision': -collision_penalty,
            'reward_time': -time_cost,
            'reward_success': success_bonus
        }
        
        if collision_count > 0 and self.current_step % 10 == 0:
            print(f"⚠️ Step {self.current_step}: 检测到 {collision_count} 次碰撞！")
        
        return float(total_reward), bool(success), reward_terms
    
    def _get_observation(self):
        """获取观察"""
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        ee_position = self._get_end_effector_position()
        target_velocity = self._estimate_target_velocity()
        
        observation = np.concatenate([
            joint_positions,
            joint_velocities,
            ee_position,
            self.target_position,
            target_velocity
        ]).astype(np.float32)
        
        return observation
    
    def _create_target_visual(self):
        """创建目标视觉"""
        if hasattr(self, 'target_visual_id'):
            try:
                p.removeBody(self.target_visual_id, physicsClientId=self.physics_client)
            except:
                pass
        
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.02,  # 增大到2cm，更明显
            rgbaColor=[1, 0.5, 0, 1.0],
            physicsClientId=self.physics_client
        )
        
        self.target_visual_id = p.createMultiBody(
            baseMass=0, 
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_position,
            physicsClientId=self.physics_client
        )
    
    def _update_target_visual(self):
        """更新目标视觉"""
        if hasattr(self, 'target_visual_id') and self.target_visual_id is not None:
            try:
                p.resetBasePositionAndOrientation(
                    self.target_visual_id, 
                    self.target_position,
                    [0, 0, 0, 1], 
                    physicsClientId=self.physics_client
                )
            except:
                pass
    
    def _randomize_dynamics(self):
        """领域随机化"""
        if not self.enable_domain_randomization:
            return
        
        # 根据课程阶段调整强度
        if self.enable_curriculum:
            if self.curriculum_stage == 0:
                strength = 0.3
            elif self.curriculum_stage == 1:
                strength = 0.6
            else:
                strength = 1.0
            self.randomization_strength = strength
        
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        # 1. 质量随机化
        mass_range_width = 0.2 * self.randomization_strength
        mass_scale = np.random.uniform(1.0 - mass_range_width, 1.0 + mass_range_width)
        
        for link_idx in self.link_indices:
            try:
                base_mass = self.link_masses.get(link_idx, 0.1)
                random_mass = base_mass * mass_scale
                p.changeDynamics(
                    self.robot_id, link_idx,
                    mass=random_mass,
                    physicsClientId=self.physics_client
                )
            except:
                pass
        
        # 2. 摩擦随机化
        friction_min = 0.05 + (0.1 - 0.05) * (1 - self.randomization_strength)
        friction_max = 0.1 + (0.2 - 0.1) * self.randomization_strength
        friction_coef = np.random.uniform(friction_min, friction_max)
        
        for link_idx in range(-1, num_joints):
            try:
                p.changeDynamics(
                    self.robot_id, link_idx,
                    lateralFriction=friction_coef,
                    physicsClientId=self.physics_client
                )
            except:
                pass
        
        # 3. 阻尼随机化
        damping_range_width = 0.75 * self.randomization_strength
        linear_damping = np.random.uniform(2.0 - damping_range_width, 2.0 + damping_range_width)
        angular_damping = np.random.uniform(2.0 - damping_range_width, 2.0 + damping_range_width)
        
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
        
        # 4. 水流随机化
        current_range_width = 0.15 * self.randomization_strength
        self.current_velocity = np.random.uniform(
            -current_range_width,
            current_range_width,
            size=3
        )
        
        # 5. 重力随机化
        gravity_scale = np.random.uniform(0.95, 1.05)
        randomized_gravity = self.gravity * gravity_scale * 0.1
        p.setGravity(0, 0, randomized_gravity, physicsClientId=self.physics_client)
        
        # 6. 控制延迟随机化
        if self.enable_control_delay:
            self.current_control_delay = np.random.uniform(*self.control_delay_range)
        
        # 记录参数
        self.current_randomization = {
            'mass_scale': mass_scale,
            'friction': friction_coef,
            'damping': (linear_damping, angular_damping),
            'current_velocity': self.current_velocity.copy(),
            'gravity_scale': gravity_scale,
            'strength': self.randomization_strength
        }
        
        # 每100个episode打印一次
        if self.episode_count % 100 == 0:
            print(f"\nEpisode {self.episode_count}:")
            print(f"  课程阶段: {self.curriculum_stage}")
            print(f"  随机化强度: {self.randomization_strength:.2f}")
            print(f"  目标漂移限制: {self.max_target_drift*100:.0f}cm")
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        if self.physics_client is None:
            self._connect_physics()
            self._setup_underwater_scene()
            self._analyze_robot()
        
        self.current_step = 0
        
        # Episode计数
        if not hasattr(self, 'episode_count'):
            self.episode_count = 0
        self.episode_count += 1
        
        # ✅ 更新课程阶段
        if self.enable_curriculum:
            if self.episode_count < self.curriculum_thresholds[0]:
                self.curriculum_stage = 0
            elif self.episode_count < self.curriculum_thresholds[1]:
                self.curriculum_stage = 1
            else:
                self.curriculum_stage = 2
        
        # 随机初始化关节
        init_positions = []
        for joint_idx in self.main_joint_indices:
            joint = self.joint_info[joint_idx]
            range_center = (joint['lower'] + joint['upper']) / 2
            range_width = (joint['upper'] - joint['lower']) * 0.8
            init_pos = np.random.uniform(range_center - range_width, range_center + range_width)
            init_positions.append(init_pos)
        
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.resetJointState(
                self.robot_id, joint_idx, init_positions[i],
                physicsClientId=self.physics_client
            )
        
        # 重置夹爪
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            
            if joint_name == 'joint_5':
                p.resetJointState(self.robot_id, i, 0.0014, physicsClientId=self.physics_client)
            elif 'jaw' in joint_name.lower():
                home_angle = 0.0014 * 41.67
                p.resetJointState(self.robot_id, i, home_angle, physicsClientId=self.physics_client)
        
        # 清空缓冲区
        if hasattr(self, 'action_buffer'):
            self.action_buffer.clear()
        
        # 应用随机化
        self._randomize_dynamics()
        
        # ✅ 根据课程阶段设置目标漂移参数
        if self.enable_curriculum:
            if self.curriculum_stage == 0:
                self.max_target_drift = 0.03       # 5cm
                self.drift_damping = 0.94
                self.drift_noise_strength = 0.02
            elif self.curriculum_stage == 1:
                self.max_target_drift = 0.05       # 12cm
                self.drift_damping = 0.9
                self.drift_noise_strength = 0.05
            else:
                self.max_target_drift = 0.08       # 20cm
                self.drift_damping = 0.85
                self.drift_noise_strength = 0.08
        
        # 生成目标位置
        self.target_position = self._sample_target_position()
        self.initial_target_position = self.target_position.copy()
        self.target_velocity = np.zeros(3, dtype=np.float32)
        
        # 重置历史
        self.target_position_history = [self.target_position.copy()]
        
        # 重置水流
        self.current_velocity_actual = self.current_velocity.copy()
        
        # 创建目标视觉
        self._create_target_visual()
        
        # 稳定仿真
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # 计算初始距离
        self.previous_distance = float(np.linalg.norm(
            self._get_end_effector_position() - self.target_position
        ))
        
        observation = self._get_observation()
        
        info = {
            'target_position': self.target_position.copy(),
            'initial_distance': self.previous_distance,
            'episode_count': self.episode_count,
            'curriculum_stage': self.curriculum_stage,
            'max_drift': self.max_target_drift,
            'randomization_strength': self.randomization_strength
        }
        
        return observation, info
    
    def step(self, action):
        """执行一步"""
        self.current_step += 1
        action = np.array(action, dtype=np.float32)
        
        self._apply_action(action)
        self._apply_underwater_forces()
        
        # 16个物理子步
        for _ in range(16):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # 更新目标位置（带漂移）
        self._update_target_position(dt=16./240.)
        self._update_target_visual()
        
        # 更新历史
        self.target_position_history.append(self.target_position.copy())
        if len(self.target_position_history) > self.history_length + 1:
            self.target_position_history.pop(0)
        
        observation = self._get_observation()
        reward, success, reward_terms = self._compute_reward(action)
        
        terminated = bool(success)
        truncated = bool(self.current_step >= self.max_steps)
        
        ee_pos = self._get_end_effector_position()
        current_distance = float(np.linalg.norm(ee_pos - self.target_position))
        
        current_drift = float(np.linalg.norm(
            self.target_position - self.initial_target_position
        ))
        
        info = {
            'success': success,
            'distance': current_distance,
            'is_success': success,
            'current_velocity': self.current_velocity_actual.copy(),
            'underwater': True,
            'target_drift': current_drift,
            'target_velocity': self.target_velocity.copy(),
            'target_velocity_estimated': self._estimate_target_velocity().copy(),
            'curriculum_stage': self.curriculum_stage,
            'collision_count': reward_terms.get('reward_collision', 0) / -10.0
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
    print("水下Alpha机械臂环境测试 - 强漂移版")
    print("="*70 + "\n")
    
    env = AlphaReachEnv(render_mode="human")
    obs, info = env.reset()
    
    print(f"✅ 观察维度: {obs.shape}")
    print(f"初始目标位置: {info['target_position']}")
    print(f"最大漂移限制: {info['max_drift']*100:.0f}cm")
    print(f"课程阶段: {info['curriculum_stage']}\n")
    
    max_drift_observed = 0.0
    
    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if 'target_drift' in info:
            max_drift_observed = max(max_drift_observed, info['target_drift'])
        
        if step % 50 == 0:
            drift_cm = info.get('target_drift', 0) * 100
            target_vel_est = info.get('target_velocity_estimated', np.zeros(3))
            vel_mag = np.linalg.norm(target_vel_est)
            
            print(f"步数 {step}:")
            print(f"  距离: {info['distance']:.3f}m")
            print(f"  目标漂移: {drift_cm:.1f}cm / {env.max_target_drift*100:.0f}cm")
            print(f"  目标速度: {vel_mag*100:.1f}cm/s")
            
            if step == 50:
                print(f"  奖励分解:")
                for key in ['reward_distance', 'reward_prediction', 'reward_velocity_match', 
                           'reward_progress_vel', 'reward_collision']:
                    if key in info:
                        print(f"    {key}: {info[key]:.3f}")
        
        if done:
            print(f"\n{'='*70}")
            print(f"回合结束:")
            print(f"  最大漂移: {max_drift_observed*100:.1f}cm / {env.max_target_drift*100:.0f}cm")
            print(f"  成功: {info['success']}")
            print(f"  最终距离: {info['distance']:.3f}m")
            print(f"  碰撞次数: {int(info.get('collision_count', 0))}")
            print(f"{'='*70}\n")
            
            obs, info = env.reset()
            max_drift_observed = 0.0
    
    env.close()
    print("\n测试完成!")