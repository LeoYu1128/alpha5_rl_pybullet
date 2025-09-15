import pybullet as p
import pybullet_data  # 添加這個import
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

class AlphaRobotRL(gym.Env):
    """Alpha Robot 4-DOF 強化學習環境 - 基於400mm工作空間的目標夾取任務"""
    
    def __init__(self, render_mode="human", max_steps=500):
        super().__init__()
        self.height_offset = 0.1  # 機器人基座高度
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # 基於你提供限制的關節參數 (前4個關節)
        self.joint_limits = {
            'lower': np.array([0.0, -3.49, 0.0, 0.0]),       
            'upper': np.array([6.10, 3.49, 3.22, 3.22]),
            'max_torque': np.array([9.0, 9.0, 9.0, 9.0])  # 基於官方數據估算
        }
        
        # 水下動力學參數
        self.water_density = 1025.0
        self.damping_coeff = np.array([2.8, 2.8, 2.8, 2.8])
        
        # 漸進式訓練 - 從你的代碼借鑒
        self.torque_scale = 0.7  # 開始用30%扭矩
        self.training_stage = 1
        
        # 基於400mm工作空間的目標列表 - 確保都在可達範圍內
        self.targets = [
            np.array([0.25, 0.0, 0.15 + self.height_offset]),    # 目標1：正前方250mm
            np.array([0.20, 0.15, 0.20 + self.height_offset]),   # 目標2：右前方，距離約250mm
            np.array([0.30, -0.10, 0.18 + self.height_offset]),  # 目標3：左前方，距離約320mm
            np.array([0.05, 0.1, 0.25 + self.height_offset])     # 目標4：遠距離350mm (接近極限)
        ]
        self.current_target_idx = 0
        self.success_count = 0
        self.required_successes = 5
        
        # 連接PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
            p.resetDebugVisualizerCamera(1.0, 45, -30, [0, 0, 0.3])
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        # 設置PyBullet數據路徑 - 修正錯誤的關鍵
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        
        self._setup_scene()
        
        # 動作空間：4個關節的扭矩 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 觀察空間：[關節位置(4), 關節速度(4), 末端位置(3), 目標位置(3)] = 14維
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        
        print(f"Alpha Robot RL環境初始化完成")
        print(f"工作空間: 400mm, 目標數量: {len(self.targets)}")
        print(f"初始扭矩縮放: {self.torque_scale:.1%}")
        
    def _setup_scene(self):
        """設置場景"""
        # 加載地面 - 現在應該能找到文件
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 加載機器人 - 嘗試多個可能的路徑
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
                        useFixedBase=True
                    )
                    print(f"成功加載機器人URDF: {robot_path}")
                    break
                except Exception as e:
                    print(f"嘗試加載 {robot_path} 失敗: {e}")
                    continue
        
        if self.robot_id is None:
            raise FileNotFoundError(
                "找不到Alpha機器人URDF文件。請確保以下文件之一存在:\n" +
                "\n".join(robot_paths)
            )
        
        # 獲取前4個可控關節
        self.joint_indices = [2, 3, 4, 5]  # joint_1 到 joint_4
        
        # 設置水下阻尼
        for i, joint_idx in enumerate(self.joint_indices):
            p.changeDynamics(
                self.robot_id, joint_idx,
                jointDamping=self.damping_coeff[i],
                lateralFriction=0.5
            )
        
        # TCP索引
        self.tcp_index = 11
        
        # 創建目標視覺化
        self._create_target()
        
        # # 驗證所有目標都在400mm範圍內
        # self._validate_targets()

        
    def _create_target(self):
        """創建目標"""
        target_visual = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.025, 
            rgbaColor=[1, 0, 0, 0.8]
        )
        
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=self.targets[0]
        )
        
    # def _validate_targets(self):
    #     """驗證所有目標都在400mm工作空間內"""
    #     print("驗證目標位置:")
    #     for i, target in enumerate(self.targets):
    #         distance = np.linalg.norm(target[:2])  # 水平距離
    #         total_distance = np.linalg.norm(target)  # 總距離
    #         print(f"目標{i+1}: {target}, 水平距離: {distance*1000:.0f}mm, 總距離: {total_distance*1000:.0f}mm")
            
    #         if distance > 0.4:  # 超過400mm
    #             print(f"警告: 目標{i+1}超出400mm工作空間!")
        
    def _get_end_effector_pos(self):
        """獲取末端執行器位置"""
        link_state = p.getLinkState(self.robot_id, self.tcp_index)
        return np.array(link_state[0])
        
    def _apply_safety_limits(self, torques):
        """應用安全限制"""
        current_positions = self._get_joint_positions()
        current_velocities = self._get_joint_velocities()
        
        safe_torques = torques.copy()
        
        for i in range(4):
            pos = current_positions[i]
            vel = current_velocities[i]
            lower = self.joint_limits['lower'][i]
            upper = self.joint_limits['upper'][i]
            
            # 位置安全檢查
            safety_margin = 0.2
            
            if pos > upper - safety_margin:
                safe_torques[i] = min(0, torques[i])
            elif pos < lower + safety_margin:
                safe_torques[i] = max(0, torques[i])
                
            # 速度安全檢查
            if abs(vel) > 3.0:
                safe_torques[i] = -5.0 * vel
                
        return safe_torques
        
    def _get_joint_positions(self):
        """獲取關節位置"""
        positions = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            positions.append(joint_state[0])
        return np.array(positions)
        
    def _get_joint_velocities(self):
        """獲取關節速度"""
        velocities = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            velocities.append(joint_state[1])
        return np.array(velocities)
        
    def _get_observation(self):
        """獲取觀察"""
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        ee_pos = self._get_end_effector_pos()
        
        obs = np.concatenate([
            joint_positions, #4
            joint_velocities, #4
            ee_pos,           #3
            self.current_target #3
        ])
        
        return obs.astype(np.float32)
        
    def _calculate_reward(self):
        """计算密集的引导性奖励 - 改进版本"""
        ee_pos = self._get_end_effector_pos()
        current_distance = np.linalg.norm(ee_pos - self.current_target)
        
        total_reward = 0.0
        
        # 1. 基础距离奖励 - 使用指数衰减，距离越近奖励越大
        max_distance = 0.5  # 最大可能距离
        normalized_distance = min(current_distance / max_distance, 1.0)
        distance_reward = 10.0 * (1.0 - normalized_distance**2)  # 二次函数，越近奖励越高
        total_reward += distance_reward
        
        # 2. 进展奖励 - 奖励向目标移动的行为
        if hasattr(self, 'previous_distance') and self.previous_distance is not None:
            progress = self.previous_distance - current_distance
            progress_reward = progress * 50.0  # 每米进展给50分
            total_reward += progress_reward
        self.previous_distance = current_distance
        
        # 3. 分层接近奖励 - 进入不同距离区域给予奖励
        proximity_zones = [
            (0.20, 5.0),   # 20cm内，+5分
            (0.15, 10.0),  # 15cm内，+10分  
            (0.10, 20.0),  # 10cm内，+20分
            (0.05, 40.0),  # 5cm内，+40分
            (0.03, 80.0),  # 3cm内，+80分
        ]
        
        for threshold, bonus in proximity_zones:
            if current_distance < threshold:
                total_reward += bonus
                break  # 只给最高等级的奖励
        
        # 4. 成功奖励
        if current_distance < 0.03:
            total_reward += 200.0
        
        # 5. 停留奖励 - 在目标附近停留给予奖励
        if current_distance < 0.05:
            if not hasattr(self, 'steps_in_good_region'):
                self.steps_in_good_region = 0
            self.steps_in_good_region += 1
            staying_bonus = min(self.steps_in_good_region * 0.5, 10.0)  # 最多10分
            total_reward += staying_bonus
        else:
            self.steps_in_good_region = 0
        
        # 6. 平滑运动奖励 - 鼓励平滑的动作
        if hasattr(self, 'previous_action'):
            action_smoothness = -np.sum(np.square(self.current_action - self.previous_action))
            total_reward += action_smoothness * 2.0
        if hasattr(self, 'current_action'):
            self.previous_action = self.current_action.copy()
        
        # 7. 速度适应性奖励 - 距离远时鼓励快速移动，距离近时鼓励缓慢精确
        joint_velocities = self._get_joint_velocities()
        avg_velocity = np.mean(np.abs(joint_velocities))
        
        if current_distance > 0.15:  # 距离较远
            # 鼓励适度的速度
            optimal_velocity = 1.0
            velocity_reward = -abs(avg_velocity - optimal_velocity) * 2.0
        else:  # 距离较近
            # 鼓励缓慢精确的运动
            velocity_penalty = -avg_velocity * 5.0
            velocity_reward = velocity_penalty
        
        total_reward += velocity_reward
        
        # 8. 工作空间引导 - 渐进式惩罚而非硬截断
        ee_distance_from_base = np.linalg.norm(ee_pos[:2])
        if ee_distance_from_base > 0.4:  # 超出400mm
            overshoot = ee_distance_from_base - 0.4
            workspace_penalty = -overshoot * 200.0  # 超出越多惩罚越大
            total_reward += workspace_penalty
        elif ee_distance_from_base > 0.35:  # 接近边界时给予警告
            warning_penalty = -(ee_distance_from_base - 0.35) * 20.0
            total_reward += warning_penalty
        
        # 9. 方向引导奖励 - 奖励朝向目标的运动方向
        ee_to_target = self.current_target - ee_pos
        if np.linalg.norm(ee_to_target) > 0.001:  # 避免除零
            direction_to_target = ee_to_target / np.linalg.norm(ee_to_target)
            
            # 计算末端执行器的运动方向（简化版本）
            if hasattr(self, 'previous_ee_pos'):
                ee_velocity_vector = ee_pos - self.previous_ee_pos
                if np.linalg.norm(ee_velocity_vector) > 0.001:
                    ee_direction = ee_velocity_vector / np.linalg.norm(ee_velocity_vector)
                    direction_alignment = np.dot(ee_direction, direction_to_target)
                    direction_reward = direction_alignment * 10.0
                    total_reward += direction_reward
            self.previous_ee_pos = ee_pos.copy()
        
        # 10. 动作幅度适应性奖励
        action_magnitude = np.linalg.norm(self.current_action)
        if current_distance > 0.1:  # 距离远时
            # 鼓励较大的动作
            if action_magnitude < 0.3:
                total_reward -= 5.0  # 动作太小的惩罚
        else:  # 距离近时
            # 鼓励精细动作
            if action_magnitude > 0.7:
                total_reward -= action_magnitude * 10.0  # 动作太大的惩罚
            else:
                total_reward += (0.7 - action_magnitude) * 5.0  # 精细动作奖励
        
        # 11. 关节位置健康度奖励 - 避免极限位置
        joint_positions = self._get_joint_positions()
        for i, pos in enumerate(joint_positions):
            joint_range = self.joint_limits['upper'][i] - self.joint_limits['lower'][i]
            normalized_pos = (pos - self.joint_limits['lower'][i]) / joint_range
            
            # 在0.2-0.8范围内给予奖励，避免极限位置
            if 0.2 <= normalized_pos <= 0.8:
                total_reward += 1.0
            elif normalized_pos < 0.1 or normalized_pos > 0.9:
                total_reward -= 5.0  # 接近极限的惩罚
        
        # 12. 探索奖励 - 记录并奖励新的有效位置
        if not hasattr(self, 'visited_positions'):
            self.visited_positions = set()
        
        # 将位置量化到网格中
        grid_size = 0.02  # 2cm网格
        grid_pos = tuple(np.round(ee_pos / grid_size).astype(int))
        
        if grid_pos not in self.visited_positions and ee_distance_from_base <= 0.4:
            self.visited_positions.add(grid_pos)
            exploration_bonus = 2.0
            total_reward += exploration_bonus
        
        return total_reward
    def _is_success(self):
        """檢查是否成功"""
        ee_pos = self._get_end_effector_pos()
        distance = np.linalg.norm(ee_pos - self.current_target)
        return distance < 0.03
        
    def _check_safety_violation(self):
        """檢查安全違規"""
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        
        # 檢查位置限制
        for i, pos in enumerate(joint_positions):
            if pos < self.joint_limits['lower'][i] or pos > self.joint_limits['upper'][i]:
                return True
                
        # 檢查速度限制
        for vel in joint_velocities:
            if abs(vel) > 5.0:
                return True
                
        return False
        
    def _advance_training(self):
        """推進訓練難度"""
        if self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1
            self.success_count = 0
            print(f"切換到目標 {self.current_target_idx + 1}/{len(self.targets)}")
        else:
            if self.torque_scale < 1.0:
                self.torque_scale = min(1.0, self.torque_scale + 0.2)
                self.training_stage += 1
                self.current_target_idx = 0
                self.success_count = 0
                print(f"訓練升級! 階段 {self.training_stage}, 扭矩縮放: {self.torque_scale:.1%}")
            else:
                self.current_target_idx = 0
                self.success_count = 0
                print("繼續最高難度訓練")
                
    def step(self, action):
        """執行動作"""
        self.current_step += 1
        self.current_action = np.array(action)
        
        # 扭矩控制
        commanded_torques = action * self.joint_limits['max_torque'] * self.torque_scale
        safe_torques = self._apply_safety_limits(commanded_torques)
        
        # 應用扭矩
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.TORQUE_CONTROL,
                force=safe_torques[i]
            )
            
        # 仿真步進
        for _ in range(4):
            p.stepSimulation()
            
        observation = self._get_observation()
        reward = self._calculate_reward()
        
        success = self._is_success()
        safety_violation = self._check_safety_violation()
        terminated = success or safety_violation
        truncated = self.current_step >= self.max_steps
        
        # 漸進式訓練邏輯
        if success:
            self.success_count += 1
            if self.success_count >= self.required_successes:
                self._advance_training()
        elif terminated:
            self.success_count = 0
            
        info = {
            'success': success,
            'distance': np.linalg.norm(self._get_end_effector_pos() - self.current_target),
            'target_index': self.current_target_idx,
            'consecutive_successes': self.success_count,
            'torque_scale': self.torque_scale,
            'training_stage': self.training_stage,
            'safety_violation': safety_violation,
            'workspace_violation': np.linalg.norm(self._get_end_effector_pos()[:2]) > 0.4
        }
        
        return observation, reward, terminated, truncated, info
        
    def reset(self, seed=None, options=None):
        """重置環境"""
        super().reset(seed=seed)
        self.current_step = 0
        
        # 安全的初始位置 - 基於400mm工作空間調整
        safe_initial_positions = [0.0, 0.0, 0.0, 0.0]  # 讓機器人在中等伸展狀態
        
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, safe_initial_positions[i])
            
        # 設置當前目標
        self.current_target = self.targets[self.current_target_idx].copy()
        p.resetBasePositionAndOrientation(
            self.target_id, self.current_target, [0, 0, 0, 1]
        )
        
        # 穩定仿真
        for _ in range(60):
            p.stepSimulation()
            
        self.current_action = np.zeros(4)
        
        # 打印當前目標信息
        target_distance = np.linalg.norm(self.current_target)
        print(f"重置完成 - 目標{self.current_target_idx+1}: {self.current_target}, 距離: {target_distance*1000:.0f}mm")
        
        return self._get_observation(), {}
        
    def close(self):
        """關閉環境"""
        p.disconnect()


def test_environment():
    """測試環境"""
    env = AlphaRobotRL(render_mode="human")
    
    obs, _ = env.reset()
    print(f"觀察空間: {obs.shape}")
    print(f"動作空間: {env.action_space}")
    print("開始測試基於400mm工作空間的Alpha Robot RL環境")
    
    for step in range(1000):
        action = env.action_space.sample() * 0.2  # 小幅度動作
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 100 == 0:
            print(f"Step {step}: 距離 {info['distance']:.3f}m, "
                  f"目標 {info['target_index']+1}/{len(env.targets)}, "
                  f"成功次數 {info['consecutive_successes']}, "
                  f"扭矩縮放 {info['torque_scale']:.1%}, "
                  f"工作空間違規 {info['workspace_violation']}")
        
        if terminated or truncated:
            print(f"Episode結束: 成功={info['success']}, 安全違規={info['safety_violation']}")
            obs, _ = env.reset()
            
    env.close()

if __name__ == "__main__":
    test_environment()