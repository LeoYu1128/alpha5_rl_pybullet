"""
Alpha机器人手动控制测试
用于验证模型和仿真的基础功能
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class AlphaRobotManualControl:
    """Alpha机器人手动控制测试类"""
    
    def __init__(self):
        # 🔧 首先初始化所有属性
        self.control_strategies = {
            'joint_1': 'velocity',    # 位置控制有问题，用速度控制
            'joint_2': 'position',    # 位置控制正常
            'joint_3': 'position',    # 位置控制正常
            'joint_4': 'position',    # 位置控制正常
            'joint_5': 'position'     # 位置控制正常，避免力矩控制
        }
        
        # 安全的初始位置（基于诊断结果）
        self.safe_initial_positions = {
            'joint_1': 1.5,      # 避开0位置
            'joint_2': 0.5,      # 在限制范围内
            'joint_3': 1.0,
            'joint_4': 1.6,      # 中间位置
            'joint_5': 0.007     # 夹爪适中位置
        }
        
        # 上一次的目标位置（用于速度控制）
        self.prev_targets = {}
        
        # 测试结果记录
        self.test_results = {
            'joint_functionality': {},
            'reachability': [],
            'collision_issues': [],
            'stability_issues': []
        }
        
        # 连接PyBullet GUI
        self.physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
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
        
        # 加载场景
        self._setup_scene()
        
        # 创建调试滑条
        self._create_debug_sliders()
        
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
            print(f"❌ URDF文件未找到: {robot_path}")
            # 使用相对路径尝试
            robot_path = "alpha_robot_for_pybullet.urdf"
            
        try:
            robot_id = p.loadURDF(
                robot_path,
                basePosition=[0, 0, 0.02],
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION
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
        
        print("\n📋 关节信息:")
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
                    'max_force': joint_info[10],
                    'max_velocity': joint_info[11],
                    'control_mode': self.control_strategies.get(joint_name, 'position')
                }
                
                print(f"关节 {joint_name}:")
                print(f"  索引: {i}")
                print(f"  类型: {joint_info[2]}")
                print(f"  控制模式: {self.control_strategies.get(joint_name, 'position')}")
                print(f"  范围: [{joint_info[8]:.3f}, {joint_info[9]:.3f}]")
                print(f"  最大力: {joint_info[10]:.2f}")
                print()
                
        # 找到末端执行器
        self.tcp_index = 11  # 基于诊断结果
        print(f"🎯 末端执行器索引: {self.tcp_index}")
        
    def _set_safe_initial_positions(self):
        """设置安全的初始位置"""
        print("\n🔧 设置安全初始位置...")
        
        for joint_name, joint_idx in zip(self.joint_names, self.joint_indices):
            if joint_name in self.safe_initial_positions:
                initial_pos = self.safe_initial_positions[joint_name]
                
                # 确保在限制范围内
                info = self.joint_info[joint_name]
                initial_pos = max(info['lower'], min(info['upper'], initial_pos))
                
                p.resetJointState(self.robot_id, joint_idx, initial_pos)
                self.prev_targets[joint_name] = initial_pos
                print(f"  {joint_name}: {initial_pos:.3f}")
                
        # 稳定几步
        for _ in range(60):
            p.stepSimulation()
        
    def _create_debug_sliders(self):
        """创建调试滑条"""
        self.sliders = {}
        
        print("\n🎛️ 创建关节控制滑条...")
        
        for joint_name in self.joint_names:
            if joint_name in self.joint_info:
                info = self.joint_info[joint_name]
                
                # 使用安全初始值
                if joint_name in self.safe_initial_positions:
                    start_value = self.safe_initial_positions[joint_name]
                else:
                    start_value = (info['lower'] + info['upper']) / 2
                    
                # 确保在范围内
                start_value = max(info['lower'], min(info['upper'], start_value))
                
                slider_id = p.addUserDebugParameter(
                    paramName=f"{joint_name} ({info['control_mode']})",
                    rangeMin=info['lower'],
                    rangeMax=info['upper'],
                    startValue=start_value
                )
                
                self.sliders[joint_name] = {
                    'id': slider_id,
                    'index': info['index'],
                    'range': (info['lower'], info['upper']),
                    'control_mode': info['control_mode']
                }
                
        # 添加全局控制参数
        self.force_scale_slider = p.addUserDebugParameter(
            "力度缩放", 0.1, 2.0, 1.0
        )
        
        self.velocity_gain_slider = p.addUserDebugParameter(
            "速度增益(joint_1)", 0.1, 5.0, 1.0
        )
        
    def _create_target(self):
        """创建目标"""
        # 目标球体
        self.target_visual = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.01, 
            rgbaColor=[1, 0, 0, 0.8]
        )
        
        self.target_collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=0.02
        )
        
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.target_visual,
            baseCollisionShapeIndex=self.target_collision,
            basePosition=[0.2, 0.1, 0.2]
        )
        
        # 添加目标位置滑条
        self.target_sliders = {
            'x': p.addUserDebugParameter("目标X", -0.4, 0.4, 0.2),
            'y': p.addUserDebugParameter("目标Y", -0.4, 0.4, 0.1), 
            'z': p.addUserDebugParameter("目标Z", 0.05, 0.4, 0.2)
        }
        
    def run_manual_test(self):
        """运行手动测试"""
        print("\n🚀 开始修复版手动控制测试")
        print("=" * 60)
        print("📝 修复说明:")
        print("• Joint_1 使用速度控制（位置控制有问题）")
        print("• Joint_2,3,4,5 使用位置控制")
        print("• 设置了安全的初始位置")
        print("• 按 ESC 或关闭窗口退出")
        print("=" * 60)
        
        test_start_time = time.time()
        success_count = 0
        
        try:
            while True:
                # 读取滑条值并应用控制
                self._update_joint_control()
                
                # 更新目标位置
                self._update_target_position()
                
                # 进行测试检查
                self._perform_checks()
                
                # 检查成功情况
                if self._check_success():
                    success_count += 1
                
                # 步进仿真
                p.stepSimulation()
                
                # 添加延迟以便观察
                time.sleep(1./240.)
                
        except KeyboardInterrupt:
            print("\n⏹️ 手动测试终止")
            
        finally:
            test_duration = time.time() - test_start_time
            print(f"\n📊 测试时长: {test_duration:.1f} 秒")
            print(f"成功次数: {success_count}")
            self._generate_test_report()
            
    def _update_joint_control(self):
        """更新关节控制（使用不同的控制策略）"""
        force_scale = p.readUserDebugParameter(self.force_scale_slider)
        velocity_gain = p.readUserDebugParameter(self.velocity_gain_slider)
        
        for joint_name, slider_info in self.sliders.items():
            slider_value = p.readUserDebugParameter(slider_info['id'])
            joint_index = slider_info['index']
            control_mode = slider_info['control_mode']
            
            joint_info = self.joint_info[joint_name]
            max_force = joint_info['max_force'] * force_scale
            
            if control_mode == 'position':
                # 标准位置控制
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=slider_value,
                    force=max_force,
                    positionGain=0.1,      # 温和的P增益
                    velocityGain=0.1,      # 温和的D增益
                    maxVelocity=2.0        # 限制速度
                )
                
            elif control_mode == 'velocity':
                # Joint_1的速度控制（模拟位置控制）
                current_pos = p.getJointState(self.robot_id, joint_index)[0]
                target_pos = slider_value
                
                # 简单P控制器
                pos_error = target_pos - current_pos
                desired_velocity = pos_error * velocity_gain
                
                # 限制速度
                max_vel = joint_info['max_velocity']
                desired_velocity = np.clip(desired_velocity, -max_vel, max_vel)
                
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    p.VELOCITY_CONTROL,
                    targetVelocity=desired_velocity,
                    force=max_force
                )
                
                # 记录目标
                self.prev_targets[joint_name] = target_pos
                
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
        
    def _perform_checks(self):
        """执行各种检查"""
        # 获取当前状态
        joint_states = [p.getJointState(self.robot_id, idx) 
                       for idx in self.joint_indices]
        
        # 检查关节功能
        for i, (joint_name, state) in enumerate(zip(self.joint_names, joint_states)):
            pos, vel, _, torque = state
            
            # 记录关节是否响应
            if joint_name not in self.test_results['joint_functionality']:
                self.test_results['joint_functionality'][joint_name] = {
                    'responsive': abs(vel) > 0.001,
                    'within_limits': True,
                    'max_torque_observed': abs(torque)
                }
            else:
                # 更新记录
                if abs(vel) > 0.001:
                    self.test_results['joint_functionality'][joint_name]['responsive'] = True
                    
                self.test_results['joint_functionality'][joint_name]['max_torque_observed'] = max(
                    self.test_results['joint_functionality'][joint_name]['max_torque_observed'],
                    abs(torque)
                )
                
        # 检查末端执行器位置
        if hasattr(self, 'tcp_index'):
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
                
                # 记录可达性
                if distance < 0.05:  # 5cm内认为可达
                    reachability_record = {
                        'target': target_pos.copy(),
                        'achieved': ee_pos.copy(),
                        'distance': distance,
                        'timestamp': time.time()
                    }
                    
                    # 避免重复记录相同位置
                    if not any(np.allclose(r['target'], target_pos, atol=0.01) 
                              for r in self.test_results['reachability']):
                        self.test_results['reachability'].append(reachability_record)
            except:
                pass
                        
        # 检查碰撞
        contact_points = p.getContactPoints(self.robot_id)
        if contact_points:
            # 记录碰撞（排除与地面的正常接触）
            for contact in contact_points:
                if contact[2] != self.plane_id:  # 不是与地面的接触
                    collision_record = {
                        'bodyA': contact[1],
                        'bodyB': contact[2],
                        'linkA': contact[3],
                        'linkB': contact[4],
                        'timestamp': time.time()
                    }
                    self.test_results['collision_issues'].append(collision_record)
                    
    def _check_success(self):
        """检查是否成功到达目标"""
        try:
            # 获取TCP位置
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
            
            return distance < 0.05  # 5cm内算成功
            
        except:
            return False
            
    def _generate_test_report(self):
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("📊 手动控制测试报告")
        print("=" * 60)
        
        # 关节功能测试
        print("\n🔧 关节功能测试:")
        print("-" * 30)
        all_joints_ok = True
        
        for joint_name, result in self.test_results['joint_functionality'].items():
            status = "✅" if result['responsive'] else "❌"
            print(f"{status} {joint_name}: 响应性={result['responsive']}, "
                  f"最大力矩={result['max_torque_observed']:.2f}")
            if not result['responsive']:
                all_joints_ok = False
                
        if all_joints_ok:
            print("\n✅ 所有关节功能正常")
        else:
            print("\n❌ 部分关节存在问题，需要检查URDF或控制参数")
            
        # 可达性测试
        print(f"\n🎯 可达性测试:")
        print("-" * 30)
        reachable_count = len(self.test_results['reachability'])
        print(f"成功到达目标数量: {reachable_count}")
        
        if reachable_count > 0:
            print("成功到达的位置:")
            for i, record in enumerate(self.test_results['reachability'][:5]):  # 只显示前5个
                print(f"  {i+1}. 目标={record['target']}, "
                      f"误差={record['distance']:.4f}m")
            print("✅ 修复成功！机器人能够到达目标位置")
        else:
            print("❌ 未能到达任何目标位置，可能存在运动学问题")
            
        # 碰撞检测
        print(f"\n💥 碰撞检测:")
        print("-" * 30)
        collision_count = len(self.test_results['collision_issues'])
        if collision_count == 0:
            print("✅ 未检测到异常碰撞")
        else:
            print(f"⚠️  检测到 {collision_count} 次碰撞")
            
        # 总结和建议
        print(f"\n📋 总结:")
        print("-" * 30)
        
        if all_joints_ok and reachable_count > 3:
            print("✅ 机器人修复成功，可以进行RL训练")
            print("🚀 建议：更新alpha_env.py使用相同的控制策略")
        else:
            print("❌ 仍需进一步调试")
        
    def close(self):
        """关闭测试"""
        p.disconnect()


def main():
    """主函数"""
    print("🤖 Alpha机器人手动控制测试")
    print("=" * 50)
    
    try:
        tester = AlphaRobotManualControl()
        if tester.robot_id is not None:
            tester.run_manual_test()
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