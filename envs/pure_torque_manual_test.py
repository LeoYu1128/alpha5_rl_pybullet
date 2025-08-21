"""
纯力矩手动控制测试
验证所有关节的力矩控制是否可行
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class PureTorqueManualTest:
    """纯力矩手动控制测试类"""
    
    def __init__(self):
        # 连接PyBullet GUI
        self.physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.2]
        )
        
        # 设置物理环境
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 真实的力矩限制（来自URDF）
        self.max_torques = {
            'joint_1': 54.36,
            'joint_2': 54.36, 
            'joint_3': 47.112,
            'joint_4': 33.069,
            'joint_5': 28.992
        }
        
        # 安全的初始位置
        self.safe_initial_positions = {
            'joint_1': 1.5,
            'joint_2': 0.5,
            'joint_3': 1.0,
            'joint_4': 1.6,
            'joint_5': 0.007
        }
        
        # 测试结果记录
        self.test_results = {
            'joint_responses': {},
            'stability_issues': [],
            'dangerous_behaviors': [],
            'successful_targets': []
        }
        
        # 加载场景
        self._setup_scene()
        
        # 创建力矩控制滑条
        self._create_torque_sliders()
        
    def _setup_scene(self):
        """设置场景"""
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 加载机器人
        self.robot_id = self._load_robot()
        
        # 获取关节信息
        self._setup_joints()
        
        # 设置安全初始位置
        self._set_safe_initial_positions()
        
        # 创建目标球
        self._create_target()
        
    def _load_robot(self):
        """加载机器人URDF"""
        robot_path = os.path.join(os.path.dirname(__file__), 
                                 "../alpha_description/urdf/alpha_robot_for_pybullet.urdf")
        
        if not os.path.exists(robot_path):
            robot_path = "alpha_robot_for_pybullet.urdf"
            
        try:
            robot_id = p.loadURDF(
                robot_path,
                basePosition=[0, 0, 0.02],
                useFixedBase=True
            )
            print(f"✅ 机器人加载成功: {robot_path}")
            return robot_id
        except Exception as e:
            print(f"❌ 机器人加载失败: {e}")
            return None
            
    def _setup_joints(self):
        """设置关节信息"""
        self.joint_indices = []
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
        self.joint_info = {}
        
        print("\n📋 关节信息 (纯力矩控制):")
        print("-" * 50)
        
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            
            if joint_name in self.joint_names:
                self.joint_indices.append(i)
                self.joint_info[joint_name] = {
                    'index': i,
                    'type': joint_info[2],
                    'lower': joint_info[8],
                    'upper': joint_info[9],
                    'max_torque': self.max_torques[joint_name]
                }
                
                print(f"关节 {joint_name}:")
                print(f"  索引: {i}")
                print(f"  最大力矩: {self.max_torques[joint_name]:.2f} Nm")
                print(f"  范围: [{joint_info[8]:.3f}, {joint_info[9]:.3f}] rad")
                print()
                
        # TCP索引
        self.tcp_index = 11
        print(f"🎯 TCP索引: {self.tcp_index}")
        
    def _set_safe_initial_positions(self):
        """设置安全的初始位置"""
        print("\n🔧 设置安全初始位置...")
        
        for joint_name, joint_idx in zip(self.joint_names, self.joint_indices):
            initial_pos = self.safe_initial_positions[joint_name]
            p.resetJointState(self.robot_id, joint_idx, initial_pos, targetVelocity=0)
            print(f"  {joint_name}: {initial_pos:.3f} rad")
            
        # 稳定几步
        for _ in range(100):
            p.stepSimulation()
            
    def _create_torque_sliders(self):
        """创建力矩控制滑条"""
        self.torque_sliders = {}
        
        print("\n🎛️ 创建力矩控制滑条...")
        
        for joint_name in self.joint_names:
            max_torque = self.max_torques[joint_name]
            
            # 创建力矩滑条 [-max_torque, +max_torque]
            slider_id = p.addUserDebugParameter(
                paramName=f"{joint_name}_torque (±{max_torque:.1f}Nm)",
                rangeMin=-max_torque,
                rangeMax=max_torque,
                startValue=0.0  # 初始力矩为0
            )
            
            self.torque_sliders[joint_name] = {
                'id': slider_id,
                'index': self.joint_info[joint_name]['index'],
                'max_torque': max_torque
            }
            
        # 添加全局控制参数
        self.torque_scale_slider = p.addUserDebugParameter(
            "力矩缩放", 0.0, 1.0, 0.3
        )
        
        self.emergency_stop_slider = p.addUserDebugParameter(
            "紧急停止 (>0.5)", 0.0, 1.0, 0.0
        )
        
        # 添加目标位置滑条
        self.target_sliders = {
            'x': p.addUserDebugParameter("目标X", -0.4, 0.4, 0.2),
            'y': p.addUserDebugParameter("目标Y", -0.4, 0.4, 0.1), 
            'z': p.addUserDebugParameter("目标Z", 0.05, 0.4, 0.2)
        }
        
    def _create_target(self):
        """创建目标球"""
        target_visual = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.02, 
            rgbaColor=[1, 0, 0, 0.8]
        )
        
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=[0.2, 0.1, 0.2]
        )
        
    def run_torque_test(self):
        """运行纯力矩控制测试"""
        print("\n🚀 开始纯力矩控制测试")
        print("=" * 60)
        print("📝 测试说明:")
        print("• 所有关节都使用纯力矩控制")
        print("• 小心操作，避免机器人损坏")
        print("• 使用紧急停止滑条可立即停止所有力矩")
        print("• 观察机器人是否能稳定控制和到达目标")
        print("• 按 ESC 或关闭窗口退出")
        print("=" * 60)
        
        test_start_time = time.time()
        step_count = 0
        last_positions = None
        
        try:
            while True:
                # 检查紧急停止
                emergency_stop = p.readUserDebugParameter(self.emergency_stop_slider)
                if emergency_stop > 0.5:
                    # 紧急停止：所有关节力矩为0
                    for joint_name, slider_info in self.torque_sliders.items():
                        p.setJointMotorControl2(
                            self.robot_id,
                            slider_info['index'],
                            p.TORQUE_CONTROL,
                            force=0.0
                        )
                    print("🛑 紧急停止激活")
                else:
                    # 正常力矩控制
                    self._apply_torque_control()
                
                # 更新目标位置
                self._update_target_position()
                
                # 进行安全检查
                if step_count % 60 == 0:  # 每0.25秒检查一次
                    self._perform_safety_checks(step_count)
                
                # 记录测试数据
                if step_count % 240 == 0:  # 每秒记录一次
                    self._record_test_data(step_count)
                
                # 步进仿真
                p.stepSimulation()
                step_count += 1
                time.sleep(1./240.)
                
        except KeyboardInterrupt:
            print("\n⏹️ 纯力矩测试终止")
            
        finally:
            test_duration = time.time() - test_start_time
            print(f"\n📊 测试时长: {test_duration:.1f} 秒")
            self._generate_torque_test_report()
            
    def _apply_torque_control(self):
        """应用纯力矩控制"""
        torque_scale = p.readUserDebugParameter(self.torque_scale_slider)
        
        applied_torques = []
        
        for joint_name, slider_info in self.torque_sliders.items():
            # 读取滑条值
            raw_torque = p.readUserDebugParameter(slider_info['id'])
            
            # 应用缩放
            scaled_torque = raw_torque * torque_scale
            
            # 安全限制（额外保护）
            max_torque = slider_info['max_torque']
            safe_torque = np.clip(scaled_torque, -max_torque * 0.8, max_torque * 0.8)
            
            # 应用力矩控制
            p.setJointMotorControl2(
                self.robot_id,
                slider_info['index'],
                p.TORQUE_CONTROL,
                force=safe_torque
            )
            
            applied_torques.append(safe_torque)
            
        return applied_torques
        
    def _update_target_position(self):
        """更新目标位置"""
        target_pos = [
            p.readUserDebugParameter(self.target_sliders['x']),
            p.readUserDebugParameter(self.target_sliders['y']),
            p.readUserDebugParameter(self.target_sliders['z'])
        ]
        
        p.resetBasePositionAndOrientation(
            self.target_id, target_pos, [0, 0, 0, 1]
        )
        
    def _perform_safety_checks(self, step_count):
        """执行安全检查"""
        # 检查关节位置是否在安全范围内
        current_positions = self._get_joint_positions()
        current_velocities = self._get_joint_velocities()
        
        for i, (joint_name, pos, vel) in enumerate(zip(self.joint_names, current_positions, current_velocities)):
            joint_info = self.joint_info[joint_name]
            
            # 检查位置限制
            if pos < joint_info['lower'] or pos > joint_info['upper']:
                warning = f"⚠️  {joint_name} 超出位置限制: {pos:.3f}"
                print(warning)
                self.test_results['stability_issues'].append({
                    'time': step_count,
                    'joint': joint_name,
                    'issue': 'position_limit',
                    'value': pos
                })
                
            # 检查速度
            if abs(vel) > 5.0:  # 5 rad/s 是很快的速度
                warning = f"⚠️  {joint_name} 速度过快: {vel:.3f} rad/s"
                print(warning)
                self.test_results['stability_issues'].append({
                    'time': step_count,
                    'joint': joint_name,
                    'issue': 'high_velocity',
                    'value': vel
                })
                
    def _record_test_data(self, step_count):
        """记录测试数据"""
        # 获取当前状态
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        
        # 获取末端执行器位置
        try:
            ee_state = p.getLinkState(self.robot_id, self.tcp_index)
            ee_pos = np.array(ee_state[0])
            
            # 获取目标位置
            target_pos = np.array([
                p.readUserDebugParameter(self.target_sliders['x']),
                p.readUserDebugParameter(self.target_sliders['y']),
                p.readUserDebugParameter(self.target_sliders['z'])
            ])
            
            # 计算距离
            distance = np.linalg.norm(ee_pos - target_pos)
            
            # 检查是否成功到达目标
            if distance < 0.05:  # 5cm内
                success_record = {
                    'time': step_count,
                    'target': target_pos.copy(),
                    'achieved': ee_pos.copy(),
                    'distance': distance
                }
                
                # 避免重复记录
                if not any(np.allclose(r['target'], target_pos, atol=0.02) 
                          for r in self.test_results['successful_targets']):
                    self.test_results['successful_targets'].append(success_record)
                    print(f"✅ 到达目标! 距离: {distance:.4f}m")
                    
            # 每10秒打印一次状态
            if step_count % 2400 == 0:
                print(f"时间: {step_count/240:.1f}s, 到目标距离: {distance:.4f}m")
                
        except:
            pass
            
    def _get_joint_positions(self):
        """获取关节位置"""
        positions = []
        for joint_idx in self.joint_indices:
            pos, _, _, _ = p.getJointState(self.robot_id, joint_idx)
            positions.append(pos)
        return np.array(positions)
        
    def _get_joint_velocities(self):
        """获取关节速度"""
        velocities = []
        for joint_idx in self.joint_indices:
            _, vel, _, _ = p.getJointState(self.robot_id, joint_idx)
            velocities.append(vel)
        return np.array(velocities)
        
    def _generate_torque_test_report(self):
        """生成力矩测试报告"""
        print("\n" + "=" * 60)
        print("📊 纯力矩控制测试报告")
        print("=" * 60)
        
        # 成功目标统计
        success_count = len(self.test_results['successful_targets'])
        print(f"\n🎯 目标到达测试:")
        print(f"成功到达目标数量: {success_count}")
        
        if success_count > 0:
            print("✅ 纯力矩控制可以到达目标位置!")
            print("成功位置示例:")
            for i, record in enumerate(self.test_results['successful_targets'][:3]):
                print(f"  {i+1}. 目标: {record['target']}, 误差: {record['distance']:.4f}m")
        else:
            print("❌ 未能通过纯力矩控制到达任何目标位置")
            
        # 稳定性问题统计
        stability_issues = len(self.test_results['stability_issues'])
        print(f"\n⚠️  稳定性问题:")
        print(f"检测到问题数量: {stability_issues}")
        
        if stability_issues > 0:
            issue_types = {}
            for issue in self.test_results['stability_issues']:
                issue_type = issue['issue']
                if issue_type not in issue_types:
                    issue_types[issue_type] = 0
                issue_types[issue_type] += 1
                
            for issue_type, count in issue_types.items():
                print(f"  {issue_type}: {count} 次")
        else:
            print("✅ 未检测到稳定性问题")
            
        # 总结和建议
        print(f"\n📋 总结:")
        print("-" * 30)
        
        if success_count > 0 and stability_issues < 5:
            print("✅ 纯力矩控制测试成功!")
            print("🚀 建议: 可以实现纯力矩的端到端RL环境")
            print("💡 注意: 需要合适的力矩限制和安全机制")
        elif success_count > 0:
            print("⚠️  纯力矩控制基本可行，但需要优化稳定性")
            print("🔧 建议: 降低力矩限制或增加阻尼")
        else:
            print("❌ 纯力矩控制有问题")
            print("🔧 建议: 检查力矩范围设置或考虑混合控制")
            
        print(f"\n💡 RL环境设计建议:")
        if success_count > 0:
            print("  • 使用纯力矩控制作为动作空间")
            print("  • 设置适当的力矩限制（当前max_torque * 0.8）")
            print("  • 添加位置和速度的软约束奖励")
            print("  • 使用渐进式训练（从小力矩开始）")
        else:
            print("  • 考虑混合控制策略")
            print("  • 或者降低力矩限制进行更保守的训练")
        
    def close(self):
        """关闭测试"""
        p.disconnect()


def main():
    """主函数"""
    print("🤖 Alpha机器人纯力矩控制测试")
    print("=" * 50)
    print("⚠️  注意: 这是纯力矩控制，请小心操作!")
    print("💡 建议: 先使用小的力矩缩放值进行测试")
    
    try:
        tester = PureTorqueManualTest()
        if tester.robot_id is not None:
            tester.run_torque_test()
        else:
            print("❌ 机器人加载失败，无法进行测试")
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            tester.close()
        except:
            pass


if __name__ == "__main__":
    main()