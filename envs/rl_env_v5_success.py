import os
# 设置环境变量避免OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

class AlphaReachEnv(gym.Env):
    """
    水下Alpha机械臂到达任务环境
    任务：在水下环境中控制4个主要关节，让末端执行器到达目标位置
    考虑流体阻力、浮力、水流干扰等水下物理特性
    """
    
    def __init__(self, render_mode=None, max_steps=500, reward_type='dense'):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.physics_client = None
        self.reward_type = reward_type  # 'dense' 或 'sparse'
        
        # 水下环境参数
        self.water_density = 1000.0  # 水密度 kg/m³
        self.gravity = -9.81  # 重力加速度
        self.base_height = 0.1
        self.workspace_radius = 0.4
        
        # 流体动力学参数
        self.drag_coefficient = 0.5  # 阻力系数
        self.added_mass_coefficient = 0.3  # 附加质量系数
        self.buoyancy_enabled = True  # 是否启用浮力
        
        # 水流参数
        self.current_velocity = np.array([0.1, 0.05, 0.0])  # 水流速度 m/s
        self.current_variation = True  # 水流是否变化
        self.turbulence_strength = 0.02  # 湍流强度
        
        # 连接物理引擎
        self._connect_physics()
        self._setup_underwater_scene()
        self._analyze_robot()

        # 奖励 shaping 系数
        self.distance_scale = 1.0
        self.progress_scale = 1.2
        self.control_penalty = 0.01
        self.velocity_penalty = 0.02
        self.time_penalty = 0.005
        self.success_bonus = 5.0
        self.success_threshold = 0.05
        self.previous_distance = None

        # 定义动作空间 - 4个主要关节的位置增量 (归一化到[-1,1])
        self.action_space = spaces.Box(
            low=-1.0,   # 归一化动作空间
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # 定义观察空间 - [4个关节位置, 4个关节速度, 3个末端位置, 3个目标位置, 3个水流速度]
        obs_dim = 4 + 4 + 3 + 3 + 3  # 17维 (增加水流信息)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        print("水下Alpha机械臂环境初始化完成")
        print(f"动作空间: {self.action_space}")
        print(f"观察空间: {self.observation_space.shape}")
        print(f"水流速度: {self.current_velocity}")
    
    def _connect_physics(self):
        """连接PyBullet物理引擎"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
        
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(
                p.COV_ENABLE_MOUSE_PICKING, 0, 
                physicsClientId=self.physics_client
            )
            # 设置水下视角
            p.resetDebugVisualizerCamera(
                2.0, 30, -20, [0, 0, 0.3], 
                physicsClientId=self.physics_client
            )
            # 设置蓝色背景模拟水下环境
            p.configureDebugVisualizer(
                p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1,
                physicsClientId=self.physics_client
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        # 设置灰色背景（代替白色）
        p.configureDebugVisualizer(
            p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1,
            physicsClientId=self.physics_client
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, 1,
            physicsClientId=self.physics_client
        )

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        
        # 设置重力 (水下环境重力影响较小)
        p.setGravity(0, 0, self.gravity * 0.1, physicsClientId=self.physics_client)  # 减弱重力影响
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
    
    def _setup_underwater_scene(self):
        """设置水下仿真场景"""
        # 加载海底地面
        self.plane_id = p.loadURDF(
            "plane.urdf", 
            physicsClientId=self.physics_client
        )
        
        # 将地面设置为深蓝色模拟海底
        p.changeVisualShape(
            self.plane_id, -1, 
            rgbaColor=[0.1, 0.2, 0.4, 1.0],
            physicsClientId=self.physics_client
        )
        
        # 加载Alpha机械臂
        # 兼容不同工作目录与GUI/DIRECT模式的URDF查找
        try:
            from pathlib import Path
            this_dir = Path(__file__).resolve().parent
            proj_root = this_dir.parent  # 项目根目录（包含 alpha_description/）
        except Exception:
            this_dir = None
            proj_root = None

        robot_paths = [
            # 相对路径候选
            "alpha_robot_for_pybullet.urdf",
            "alpha_description/urdf/alpha_robot_for_pybullet.urdf",
            "../alpha_description/urdf/alpha_robot_for_pybullet.urdf",
        ]
        # 绝对路径候选（基于当前文件与项目根）
        if proj_root is not None:
            robot_paths.extend([
                str((proj_root / "alpha_description/urdf/alpha_robot_for_pybullet.urdf").resolve()),
                str((proj_root.parent / "alpha_description/urdf/alpha_robot_for_pybullet.urdf").resolve()),
            ])
        
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
                    print(f"成功加载水下Alpha机械臂: {robot_path}")
                    break
                except Exception as e:
                    print(f"加载{robot_path}失败: {e}")
                    continue
        
        if self.robot_id is None:
            raise FileNotFoundError("找不到Alpha机械臂URDF文件")
        
        # 添加一些水下装饰物
        self._add_underwater_decorations()

    def _add_underwater_decorations(self):
        """添加水下装饰物"""
        try:
            # 添加一些球体作为水下障碍物/装饰
            for i in range(3):
                x = np.random.uniform(-0.8, 0.8)
                y = np.random.uniform(-0.8, 0.8)
                z = np.random.uniform(0.1, 0.5)
                
                decoration_id = p.loadURDF(
                    "sphere_small.urdf",
                    basePosition=[x, y, z],
                    physicsClientId=self.physics_client
                )
                
                # 设置为深绿色模拟海藻或礁石
                p.changeVisualShape(
                    decoration_id, -1, 
                    rgbaColor=[0.1, 0.4, 0.2, 0.8],
                    physicsClientId=self.physics_client
                )
        except:
            print("无法加载装饰物，跳过...")
    
    def _analyze_robot(self):
        """分析机器人结构，提取关节信息并设置水下物理特性"""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        # 存储关节信息
        self.joint_info = {}
        self.main_joint_indices = []
        
        # 遍历所有关节
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
            
            # 识别主要控制关节
            if joint_name in ['joint_1', 'joint_2', 'joint_3', 'joint_4']:
                self.main_joint_indices.append(i)
        
        print(f"找到主要控制关节: {len(self.main_joint_indices)}个")
        for i, joint_idx in enumerate(self.main_joint_indices):
            joint = self.joint_info[joint_idx]
            print(f"  [{i}] {joint['name']}: [{joint['lower']:.2f}, {joint['upper']:.2f}] 弧度")
        
        # 找到末端执行器
        self.tcp_index = None
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            if 'tcp' in joint_name.lower():
                self.tcp_index = i
                break
        
        if self.tcp_index is None:
            self.tcp_index = num_joints - 1
        
        print(f"末端执行器索引: {self.tcp_index}")
        
        # 设置水下物理特性
        self._setup_underwater_dynamics()
    
    def _setup_underwater_dynamics(self):
        """设置水下动力学特性"""
        # 为每个link设置水下物理特性
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        for i in range(-1, num_joints):  # -1代表base link
            # 增加阻尼模拟水阻
            if i >= 0:
                p.changeDynamics(
                    self.robot_id, i,
                    linearDamping=2.0,     # 线性阻尼 (水阻)
                    angularDamping=2.0,    # 角阻尼
                    jointDamping=0.5,      # 关节阻尼
                    physicsClientId=self.physics_client
                )
            else:
                # Base link
                p.changeDynamics(
                    self.robot_id, i,
                    linearDamping=1.0,
                    angularDamping=1.0,
                    physicsClientId=self.physics_client
                )
            
            # 如果启用浮力，修改质量和密度
            if self.buoyancy_enabled:
                mass_info = p.getDynamicsInfo(self.robot_id, i, physicsClientId=self.physics_client)
                original_mass = mass_info[0]
                
                # 模拟浮力效果：减少有效质量
                buoyancy_factor = 0.7  # 浮力抵消70%的重量
                effective_mass = original_mass * buoyancy_factor
                
                if effective_mass > 0:
                    p.changeDynamics(
                        self.robot_id, i,
                        mass=effective_mass,
                        physicsClientId=self.physics_client
                    )
    
    def _update_current_velocity(self):
        """更新水流速度 (模拟变化的水流)"""
        if self.current_variation:
            # 添加时变水流和湍流
            time_factor = self.current_step * 0.01
            
            # 基础水流 + 周期性变化 + 随机湍流
            base_current = self.current_velocity.copy()
            periodic_variation = np.array([
                0.03 * np.sin(time_factor),
                0.02 * np.cos(time_factor * 1.5),
                0.01 * np.sin(time_factor * 0.5)
            ])
            turbulence = np.random.normal(0, self.turbulence_strength, 3)
            
            self.current_velocity_actual = base_current + periodic_variation + turbulence
        else:
            self.current_velocity_actual = self.current_velocity.copy()
    
    def _apply_underwater_forces(self):
        """应用水下力：水流、阻力等"""
        self._update_current_velocity()
        
        # 获取末端执行器状态
        ee_state = p.getLinkState(
            self.robot_id, self.tcp_index, 
            computeLinkVelocity=1,
            physicsClientId=self.physics_client
        )
        ee_velocity = np.array(ee_state[6])  # 线速度
        
        # 计算相对于水流的速度
        relative_velocity = ee_velocity - self.current_velocity_actual
        
        # 应用流体阻力 F = -0.5 * ρ * Cd * A * v * |v|
        drag_force = -0.5 * self.water_density * self.drag_coefficient * 0.01 * \
                     relative_velocity * np.linalg.norm(relative_velocity)
        
        # 限制力的大小避免不稳定
        max_force = 8.0
        drag_force = np.clip(drag_force, -max_force, max_force)
        
        # 应用阻力到末端执行器
        p.applyExternalForce(
            self.robot_id, self.tcp_index,
            forceObj=drag_force,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.physics_client
        )
        
        # 对所有关节应用较小的水流力
        for joint_idx in self.main_joint_indices:
            # 获取关节位置
            joint_state = p.getJointState(self.robot_id, joint_idx, physicsClientId=self.physics_client)
            joint_velocity = joint_state[1]
            
            # 应用关节阻尼力 (模拟水阻对关节运动的影响)
            damping_torque = -0.1 * joint_velocity
            
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                p.TORQUE_CONTROL,
                force=damping_torque,
                physicsClientId=self.physics_client
            )
    
    def _sample_target_position(self):
        """在机械臂可达范围内采样目标位置"""
        max_attempts = 50

        # 更保守的可达范围设置
        safe_radius_min = 0.12       # 最小距离
        safe_radius_max = 0.25       # 最大距离 (保守估计可达范围)
        z_min = 0.15                 # 高度下限
        z_max = 0.35                 # 高度上限

        for _ in range(max_attempts):
            # 在圆柱体内采样 (更符合机械臂可达范围)
            r = np.random.uniform(safe_radius_min, safe_radius_max)
            theta = np.random.uniform(-np.pi/2, np.pi/2)  # 限制在前方扇形区域
            z = np.random.uniform(z_min, z_max)

            x = r * np.cos(theta)
            y = r * np.sin(theta)

            return np.array([x, y, z], dtype=np.float32)

        # 如果采样失败，返回一个确定可达的位置
        return np.array([0.2, 0.0, 0.25], dtype=np.float32)

    def _get_joint_positions(self):
        """获取主要关节的当前位置"""
        positions = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(
                self.robot_id, joint_idx,
                physicsClientId=self.physics_client
            )
            positions.append(joint_state[0])
        return np.array(positions, dtype=np.float32)
    
    def _get_joint_velocities(self):
        """获取主要关节的当前速度"""
        velocities = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(
                self.robot_id, joint_idx,
                physicsClientId=self.physics_client
            )
            velocities.append(joint_state[1])
        return np.array(velocities, dtype=np.float32)
    
    def _get_end_effector_position(self):
        """获取末端执行器位置"""
        link_state = p.getLinkState(
            self.robot_id, self.tcp_index,
            physicsClientId=self.physics_client
        )
        return np.array(link_state[0], dtype=np.float32)
    
    def _apply_action(self, action):
        """应用动作到水下机械臂"""
        # 获取当前关节位置
        current_positions = self._get_joint_positions()

        # 将归一化动作缩放到实际增量 (增大动作幅度)
        scaled_action = action * 0.15  # 将[-1,1]缩放到[-0.15,0.15]
        target_positions = current_positions + scaled_action
        
        # 应用关节限制
        for i, joint_idx in enumerate(self.main_joint_indices):
            joint = self.joint_info[joint_idx]
            target_positions[i] = np.clip(
                target_positions[i],
                joint['lower'],
                joint['upper']
            )
        
        # 执行位置控制 (水下环境力和速度都较小)
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                maxVelocity=1.0,  # 水下速度更慢
                force=500,        # 水下需要更大的力克服阻力
                physicsClientId=self.physics_client
            )
    
    def compute_goal_reward(self, achieved_goal, desired_goal, info=None):
        """基础距离奖励 (保留兼容性)"""
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        if hasattr(self, 'reward_type') and self.reward_type == 'sparse':
            return -(distance > self.success_threshold).astype(np.float32)

        return -self.distance_scale * distance.astype(np.float32)

    def _compute_reward(self, action):
        """组合距离、进度与代价的奖励"""
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

        total_reward = shaped_distance + progress_reward - control_cost - velocity_cost - time_cost
        success_bonus = self.success_bonus if success else 0.0
        total_reward += success_bonus

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
    
    def _get_observation(self):
        """获取水下环境观察 (包含水流信息)"""
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        ee_position = self._get_end_effector_position()
        
        # 组合观察向量 (增加水流速度信息)
        observation = np.concatenate([
            joint_positions,              # 4维
            joint_velocities,             # 4维  
            ee_position,                  # 3维
            self.target_position,         # 3维
            self.current_velocity_actual  # 3维 - 当前水流速度 ################################################################################
        ]).astype(np.float32)
        
        return observation
    
    def _create_target_visual(self):
        """创建目标位置的可视化标记 (水下风格)"""
        if hasattr(self, 'target_visual_id'):
            try:
                p.removeBody(self.target_visual_id, physicsClientId=self.physics_client)
            except:
                pass
        
        # 创建橙色球体作为目标标记 (在水下更显眼)
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.04,  # 恢复到原始尺寸的一半
            rgbaColor=[1, 0.5, 0, 1.0],  # 橙色，完全不透明
            physicsClientId=self.physics_client
        )
        
        self.target_visual_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_position,
            physicsClientId=self.physics_client
        )
    
    def reset(self, seed=None, options=None):
        """重置水下环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 确保物理连接存在
        if self.physics_client is None:
            self._connect_physics()
            self._setup_underwater_scene()
            self._analyze_robot()
        
        self.current_step = 0
        
        # 随机初始化关节位置
        init_positions = []
        for joint_idx in self.main_joint_indices:
            joint = self.joint_info[joint_idx]
            range_center = (joint['lower'] + joint['upper']) / 2
            range_width = (joint['upper'] - joint['lower']) * 0.3  # 水下初始化范围更小
            init_pos = np.random.uniform(
                range_center - range_width,
                range_center + range_width
            )
            init_positions.append(init_pos)
        
        # 设置关节初始位置
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.resetJointState(
                self.robot_id, joint_idx, init_positions[i],
                physicsClientId=self.physics_client
            )
        
        # 生成新的目标位置
        self.target_position = self._sample_target_position()

        # 重置水流
        self.current_velocity_actual = self.current_velocity.copy()

        # 创建目标可视化（始终创建，以便在GIF中可见）
        self._create_target_visual()
        
        # 稳定仿真
        for _ in range(100):  # 水下需要更长时间稳定
            p.stepSimulation(physicsClientId=self.physics_client)

        self.previous_distance = float(np.linalg.norm(
            self._get_end_effector_position() - self.target_position
        ))

        observation = self._get_observation()
        # 添加这部分
        info = {
            'target_position': self.target_position.copy(),
            'initial_distance': self.previous_distance
        }

        return observation, info  # 从 return observation 改为 return observation, info
    
    def step(self, action):
        """执行一步水下动作"""
        self.current_step += 1
        
        # 确保action类型正确
        action = np.array(action, dtype=np.float32)
        
        # 应用动作
        self._apply_action(action)
        
        # 应用水下力学效应
        self._apply_underwater_forces()
        
        # 运行物理仿真 (减少步数提高响应性)
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # 获取新的观察
        observation = self._get_observation()

        # 计算奖励
        reward, success, reward_terms = self._compute_reward(action)
        
        # 检查终止条件
        terminated = bool(success)
        truncated = bool(self.current_step >= self.max_steps)
        
        # 计算当前距离
        ee_pos = self._get_end_effector_position()
        current_distance = float(np.linalg.norm(ee_pos - self.target_position))
        
        info = {
            'success': success,
            'distance': current_distance,
            'is_success': success,
            'current_velocity': self.current_velocity_actual.copy(),
            'underwater': True
        }

        info.update(reward_terms)

        return observation, reward, terminated, truncated, info

    def close(self):
        """关闭环境"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.physics_client = None

    
    def render(self):
        """渲染水下环境"""
        pass

# 测试水下环境
if __name__ == "__main__":
    # 创建水下环境
    env = AlphaReachEnv(render_mode="human")
    
    print("水下环境测试开始...")
    
    # 重置环境
    obs = env.reset()
    print(f"初始观察维度: {obs.shape}")
    
    # 运行测试
    for step in range(200):
        # 随机动作 (幅度较小适应水下环境)
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        
        if step % 20 == 0:
            print(f"步数 {step}: 奖励={reward:.3f}, 距离={info['distance']:.3f}m, "
                  f"成功={info['success']}, 水流={np.linalg.norm(info['current_velocity']):.3f}")
        
        if done:
            print(f"回合结束: 成功={info['success']}, 最终距离={info['distance']:.3f}m")
            obs = env.reset()
    
    env.close()
    print("水下环境测试完成")
