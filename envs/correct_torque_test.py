"""
正确的力矩控制测试
使用控制算法，而不是直接人工设置力矩
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class CorrectTorqueControlTest:
    """正确的力矩控制测试"""
    
    def __init__(self):
        # 连接PyBullet
        self.physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.resetDebugVisualizerCamera(0.8, 45, -30, [0, 0, 0.2])
        
        # 设置物理环境
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 真实的力矩限制
        self.max_torques = np.array([54.36, 54.36, 47.112, 33.069, 28.992])
        
        # 安全的初始位置
        self.safe_initial_positions = np.array([1.5, 0.5, 1.0, 1.6, 0.007])
        
        # 关节限制
        self.joint_limits = {
            'lower': np.array([0.032, 0.0174533, 0.0174533, -3.14159, 0.0013]),
            'upper': np.array([6.02, 3.40339, 3.40339, 3.14159, 0.0133])
        }
        
        # 简单PID参数（用于力矩控制）
        self.Kp = np.array([30.0, 30.0, 25.0, 15.0, 100.0])  # 比例增益
        self.Kd = np.array([2.0, 2.0, 1.5, 1.0, 5.0])       # 微分增益
        self.Ki = np.array([0.1, 0.1, 0.1, 0.1, 1.0])       # 积分增益
        
        # PID历史
        self.integral_errors = np.zeros(5)
        self.prev_errors = np.zeros(5)
        
        # 加载场景
        self._setup_scene()
        
        # 创建目标位置滑条（不是力矩滑条！）
        self._create_target_sliders()
        
    def _setup_scene(self):
        """设置场景"""
        # 加载地面和机器人
        self.plane_id = p.loadURDF("plane.urdf")
        
        robot_path = os.path.join(os.path.dirname(__file__), 
                                 "../alpha_description/urdf/alpha_robot_for_pybullet.urdf")
        if not os.path.exists(robot_path):
            robot_path = "alpha_robot_for_pybullet.urdf"
            
        self.robot_id = p.loadURDF(robot_path, basePosition=[0, 0, 0.02], useFixedBase=True)
        
        # 获取关节信息
        self.joint_indices = [2, 3, 4, 5, 7]  # 基于之前的诊断
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
        self.tcp_index = 11
        
        # 设置安全初始位置
        for i, (joint_idx, initial_pos) in enumerate(zip(self.joint_indices, self.safe_initial_positions)):
            p.resetJointState(self.robot_id, joint_idx, initial_pos, targetVelocity=0)
            
        # 稳定仿真
        for _ in range(100):
            p.stepSimulation()
            
        # 创建目标球
        target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 0.8])
        self.target_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual, basePosition=[0.2, 0.1, 0.2])
        
    def _create_target_sliders(self):
        """创建目标位置滑条（不是直接力矩滑条！）"""
        print("🎛️ 创建目标位置控制滑条（力矩控制实现）...")
        
        self.target_sliders = {}
        
        # 为每个关节创建目标位置滑条
        for i, joint_name in enumerate(self.joint_names):
            lower = self.joint_limits['lower'][i]
            upper = self.joint_limits['upper'][i]
            initial = self.safe_initial_positions[i]
            
            slider_id = p.addUserDebugParameter(
                paramName=f"{joint_name}_target (torque_ctrl)",
                rangeMin=lower,
                rangeMax=upper,
                startValue=initial
            )
            
            self.target_sliders[joint_name] = slider_id
            
        # 控制参数
        self.kp_scale_slider = p.addUserDebugParameter("Kp缩放", 0.1, 3.0, 1.0)
        self.kd_scale_slider = p.addUserDebugParameter("Kd缩放", 0.1, 3.0, 1.0)
        self.torque_limit_slider = p.addUserDebugParameter("力矩限制", 0.1, 1.0, 0.5)
        
        # 末端目标
        self.target_pos_sliders = {
            'x': p.addUserDebugParameter("目标X", -0.4, 0.4, 0.2),
            'y': p.addUserDebugParameter("目标Y", -0.4, 0.4, 0.1), 
            'z': p.addUserDebugParameter("目标Z", 0.05, 0.4, 0.2)
        }
        
    def run_correct_torque_test(self):
        """运行正确的力矩控制测试"""
        print("\n🚀 开始正确的力矩控制测试")
        print("=" * 60)
        print("📝 关键区别:")
        print("• 不直接设置力矩值")
        print("• 设置目标位置，用PID算法计算力矩")
        print("• 这就是RL要学习的：observation → action(torque)")
        print("• 观察是否稳定和能到达目标")
        print("=" * 60)
        
        success_count = 0
        test_start_time = time.time()
        
        try:
            while True:
                # 🎯 关键：用控制算法计算力矩，不是直接设置！
                torques = self._compute_control_torques()
                
                # 应用力矩控制
                self._apply_torque_control(torques)
                
                # 更新目标球位置
                self._update_target_visualization()
                
                # 检查成功
                if self._check_success():
                    success_count += 1
                    
                # 打印状态
                if hasattr(self, 'step_count'):
                    self.step_count += 1
                else:
                    self.step_count = 0
                    
                if self.step_count % 240 == 0:  # 每秒打印
                    self._print_status(success_count)
                
                # 步进仿真
                p.stepSimulation()
                time.sleep(1./240.)
                
        except KeyboardInterrupt:
            print("\n⏹️ 测试终止")
            
        finally:
            test_duration = time.time() - test_start_time
            self._generate_final_report(test_duration, success_count)
            
    def _compute_control_torques(self):
        """计算控制力矩 - 这是关键！"""
        # 读取目标位置（人设置的目标）
        target_positions = np.zeros(5)
        for i, joint_name in enumerate(self.joint_names):
            target_positions[i] = p.readUserDebugParameter(self.target_sliders[joint_name])
            
        # 读取当前状态
        current_positions = np.zeros(5)
        current_velocities = np.zeros(5)
        
        for i, joint_idx in enumerate(self.joint_indices):
            pos, vel, _, _ = p.getJointState(self.robot_id, joint_idx)
            current_positions[i] = pos
            current_velocities[i] = vel
            
        # 读取控制参数
        kp_scale = p.readUserDebugParameter(self.kp_scale_slider)
        kd_scale = p.readUserDebugParameter(self.kd_scale_slider)
        torque_limit = p.readUserDebugParameter(self.torque_limit_slider)
        
        # 🎯 PID控制算法计算力矩（这就是RL要学的！）
        errors = target_positions - current_positions
        
        # 积分项
        self.integral_errors += errors * (1./240.)
        self.integral_errors = np.clip(self.integral_errors, -1.0, 1.0)  # 防止积分饱和
        
        # 微分项  
        derivative_errors = (errors - self.prev_errors) * 240.
        self.prev_errors = errors.copy()
        
        # PID输出
        torques = (self.Kp * kp_scale * errors + 
                  self.Ki * self.integral_errors + 
                  self.Kd * kd_scale * derivative_errors)
        
        # 安全限制
        max_allowed_torques = self.max_torques * torque_limit
        torques = np.clip(torques, -max_allowed_torques, max_allowed_torques)
        
        # 位置安全检查
        for i in range(5):
            pos = current_positions[i]
            vel = current_velocities[i]
            
            # 接近位置限制时减少力矩
            if pos > self.joint_limits['upper'][i] - 0.1:
                torques[i] = min(0, torques[i])  # 只允许反向力矩
            elif pos < self.joint_limits['lower'][i] + 0.1:
                torques[i] = max(0, torques[i])  # 只允许正向力矩
                
            # 速度过快时施加阻尼
            if abs(vel) > 2.0:
                torques[i] -= 5.0 * vel  # 阻尼力矩
                
        return torques
        
    def _apply_torque_control(self, torques):
        """应用力矩控制"""
        for i, (joint_idx, torque) in enumerate(zip(self.joint_indices, torques)):
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                p.TORQUE_CONTROL,
                force=torque
            )
            
    def _update_target_visualization(self):
        """更新目标球位置"""
        target_pos = [
            p.readUserDebugParameter(self.target_pos_sliders['x']),
            p.readUserDebugParameter(self.target_pos_sliders['y']),
            p.readUserDebugParameter(self.target_pos_sliders['z'])
        ]
        p.resetBasePositionAndOrientation(self.target_id, target_pos, [0, 0, 0, 1])
        
    def _check_success(self):
        """检查是否成功到达目标"""
        try:
            ee_state = p.getLinkState(self.robot_id, self.tcp_index)
            ee_pos = np.array(ee_state[0])
            
            target_pos = np.array([
                p.readUserDebugParameter(self.target_pos_sliders['x']),
                p.readUserDebugParameter(self.target_pos_sliders['y']),
                p.readUserDebugParameter(self.target_pos_sliders['z'])
            ])
            
            distance = np.linalg.norm(ee_pos - target_pos)
            return distance < 0.05
        except:
            return False
            
    def _print_status(self, success_count):
        """打印状态信息"""
        # 获取当前关节位置
        current_positions = []
        for joint_idx in self.joint_indices:
            pos, _, _, _ = p.getJointState(self.robot_id, joint_idx)
            current_positions.append(pos)
            
        # 检查是否有关节超限
        violations = 0
        for i, pos in enumerate(current_positions):
            if pos < self.joint_limits['lower'][i] or pos > self.joint_limits['upper'][i]:
                violations += 1
                
        # 获取末端位置
        try:
            ee_state = p.getLinkState(self.robot_id, self.tcp_index)
            ee_pos = ee_state[0]
            target_pos = [
                p.readUserDebugParameter(self.target_pos_sliders['x']),
                p.readUserDebugParameter(self.target_pos_sliders['y']),
                p.readUserDebugParameter(self.target_pos_sliders['z'])
            ]
            distance = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
            
            print(f"时间: {self.step_count/240:.1f}s, 成功: {success_count}, "
                  f"违规: {violations}, 距离: {distance:.4f}m")
        except:
            print(f"时间: {self.step_count/240:.1f}s, 成功: {success_count}, 违规: {violations}")
            
    def _generate_final_report(self, duration, success_count):
        """生成最终报告"""
        print("\n" + "=" * 60)
        print("📊 正确力矩控制测试报告")
        print("=" * 60)
        
        print(f"测试时长: {duration:.1f} 秒")
        print(f"成功次数: {success_count}")
        
        if success_count > 5:
            print("✅ 力矩控制测试成功!")
            print("🎯 结论: 可以用力矩控制实现位置控制")
            print("🚀 建议: RL可以学习这种控制策略")
            print("💡 关键: RL要学的是 observation → torque 的映射")
        else:
            print("❌ 力矩控制仍有问题")
            print("🔧 建议: 调整PID参数或使用混合控制")
            
        print(f"\n💭 对比理解:")
        print(f"• 之前的测试: 人直接设置力矩 → 关节飞掉")
        print(f"• 现在的测试: 算法计算力矩 → 稳定控制")
        print(f"• RL的任务: 学会像PID一样计算正确的力矩")
        
    def close(self):
        """关闭测试"""
        p.disconnect()


def main():
    """主函数"""
    print("🤖 正确的力矩控制测试")
    print("=" * 50)
    print("💡 关键理解:")
    print("• 力矩控制 ≠ 直接设置力矩值")
    print("• 力矩控制 = 用算法计算合适的力矩")
    print("• RL要学的就是这个算法!")
    
    try:
        tester = CorrectTorqueControlTest()
        tester.run_correct_torque_test()
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            tester.close()
        except:
            pass


if __name__ == "__main__":
    main()