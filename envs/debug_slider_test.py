"""
调试滑条控制问题
禁用鼠标交互，专注测试滑条
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class SliderDebugTest:
    """调试滑条控制"""
    
    def __init__(self):
        # 连接PyBullet
        self.physics_client = p.connect(p.GUI)
        
        # 🔧 关键：禁用鼠标拖动干扰！
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        
        print("✅ 已禁用鼠标拖动，现在滑条应该有用了")
        
        p.resetDebugVisualizerCamera(0.8, 45, -30, [0, 0, 0.2])
        
        # 设置物理环境
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 加载场景
        self._setup_scene()
        
        # 创建简单的滑条测试
        self._create_simple_sliders()
        
    def _setup_scene(self):
        """设置场景"""
        # 加载地面和机器人
        self.plane_id = p.loadURDF("plane.urdf")
        
        robot_path = os.path.join(os.path.dirname(__file__), 
                                 "../alpha_description/urdf/alpha_robot_for_pybullet.urdf")
        if not os.path.exists(robot_path):
            robot_path = "alpha_robot_for_pybullet.urdf"
            
        self.robot_id = p.loadURDF(robot_path, basePosition=[0, 0, 0.02], useFixedBase=True)
        print(f"✅ 机器人加载成功")
        
        # 关节信息
        self.joint_indices = [2, 3, 4, 5, 7]  # joint_1到joint_5的索引
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
        
        # 设置初始位置
        initial_positions = [1.5, 0.5, 1.0, 1.6, 0.007]
        for joint_idx, initial_pos in zip(self.joint_indices, initial_positions):
            p.resetJointState(self.robot_id, joint_idx, initial_pos)
            
        # 稳定仿真
        for _ in range(100):
            p.stepSimulation()
            
    def _create_simple_sliders(self):
        """创建简单的滑条测试"""
        print("\n🎛️ 创建调试滑条...")
        
        # 只测试一个关节，避免复杂性
        self.test_joint_idx = 0  # 测试joint_1
        self.test_joint_name = self.joint_names[0]
        self.test_pybullet_idx = self.joint_indices[0]
        
        print(f"🎯 测试关节: {self.test_joint_name} (PyBullet索引: {self.test_pybullet_idx})")
        
        # 创建一个简单的位置控制滑条
        self.position_slider = p.addUserDebugParameter(
            "joint_1_position", 
            0.5, 5.5, 1.5  # 安全范围
        )
        
        # 创建力矩控制滑条
        self.torque_slider = p.addUserDebugParameter(
            "joint_1_torque", 
            -20.0, 20.0, 0.0  # 较小的力矩范围
        )
        
        # 控制模式选择
        self.control_mode_slider = p.addUserDebugParameter(
            "控制模式(0=位置,1=力矩)", 
            0, 1, 0
        )
        
        # 调试信息显示
        self.debug_info_slider = p.addUserDebugParameter(
            "显示调试信息(>0.5)", 
            0.0, 1.0, 1.0
        )
        
        print(f"滑条创建完成:")
        print(f"  位置滑条ID: {self.position_slider}")
        print(f"  力矩滑条ID: {self.torque_slider}")
        print(f"  控制模式ID: {self.control_mode_slider}")
        
    def run_slider_debug(self):
        """运行滑条调试"""
        print("\n🚀 开始滑条调试测试")
        print("=" * 60)
        print("📝 测试说明:")
        print("• 鼠标拖动已禁用")
        print("• 只能通过滑条控制关节")
        print("• 观察滑条是否有响应")
        print("• 注意调试信息输出")
        print("=" * 60)
        
        step_count = 0
        last_slider_values = [0, 0, 0, 0]  # 记录上次滑条值
        
        try:
            while True:
                # 🔍 读取所有滑条值
                position_value = p.readUserDebugParameter(self.position_slider)
                torque_value = p.readUserDebugParameter(self.torque_slider)
                control_mode = p.readUserDebugParameter(self.control_mode_slider)
                show_debug = p.readUserDebugParameter(self.debug_info_slider)
                
                current_values = [position_value, torque_value, control_mode, show_debug]
                
                # 🔍 检查滑条是否有变化
                slider_changed = False
                for i, (curr, last) in enumerate(zip(current_values, last_slider_values)):
                    if abs(curr - last) > 0.01:
                        slider_changed = True
                        print(f"📊 滑条 {i} 变化: {last:.3f} → {curr:.3f}")
                        
                last_slider_values = current_values.copy()
                
                # 🎮 应用控制
                if control_mode < 0.5:
                    # 位置控制模式
                    p.setJointMotorControl2(
                        self.robot_id,
                        self.test_pybullet_idx,
                        p.POSITION_CONTROL,
                        targetPosition=position_value,
                        force=50.0,
                        positionGain=0.1,
                        velocityGain=0.1,
                        maxVelocity=2.0
                    )
                    control_type = "位置控制"
                    control_value = position_value
                    
                else:
                    # 力矩控制模式  
                    p.setJointMotorControl2(
                        self.robot_id,
                        self.test_pybullet_idx,
                        p.TORQUE_CONTROL,
                        force=torque_value
                    )
                    control_type = "力矩控制"
                    control_value = torque_value
                
                # 🔍 获取关节状态
                pos, vel, _, applied_torque = p.getJointState(self.robot_id, self.test_pybullet_idx)
                
                # 📊 显示调试信息
                if show_debug > 0.5 and step_count % 60 == 0:  # 每0.25秒显示一次
                    print(f"\n🔍 调试信息 (时间: {step_count/240:.1f}s):")
                    print(f"  滑条读取:")
                    print(f"    位置滑条: {position_value:.3f}")
                    print(f"    力矩滑条: {torque_value:.3f}")
                    print(f"    控制模式: {control_mode:.1f} ({control_type})")
                    print(f"  当前控制:")
                    print(f"    控制类型: {control_type}")
                    print(f"    控制值: {control_value:.3f}")
                    print(f"  关节状态:")
                    print(f"    位置: {pos:.3f} rad")
                    print(f"    速度: {vel:.3f} rad/s")
                    print(f"    力矩: {applied_torque:.3f} Nm")
                    
                    # 🎯 检查控制是否有效
                    if control_type == "位置控制":
                        error = abs(pos - position_value)
                        if error < 0.05:
                            print(f"  ✅ 位置控制有效 (误差: {error:.4f})")
                        else:
                            print(f"  ⚠️  位置控制误差较大 (误差: {error:.4f})")
                    else:
                        if abs(torque_value) > 0.1 and abs(vel) > 0.01:
                            print(f"  ✅ 力矩控制有响应 (速度: {vel:.3f})")
                        elif abs(torque_value) > 0.1:
                            print(f"  ⚠️  力矩控制无明显响应")
                        else:
                            print(f"  ℹ️  力矩值太小，无期望响应")
                            
                # 步进仿真
                p.stepSimulation()
                step_count += 1
                time.sleep(1./240.)
                
        except KeyboardInterrupt:
            print("\n⏹️ 调试测试终止")
            
        finally:
            self._generate_debug_report(step_count)
            
    def _generate_debug_report(self, total_steps):
        """生成调试报告"""
        print("\n" + "=" * 60)
        print("📊 滑条调试报告")
        print("=" * 60)
        
        # 最终读取滑条值
        position_value = p.readUserDebugParameter(self.position_slider)
        torque_value = p.readUserDebugParameter(self.torque_slider)
        control_mode = p.readUserDebugParameter(self.control_mode_slider)
        
        print(f"最终滑条值:")
        print(f"  位置滑条: {position_value:.3f}")
        print(f"  力矩滑条: {torque_value:.3f}")
        print(f"  控制模式: {control_mode:.1f}")
        
        # 最终关节状态
        pos, vel, _, applied_torque = p.getJointState(self.robot_id, self.test_pybullet_idx)
        print(f"\n最终关节状态:")
        print(f"  位置: {pos:.3f} rad")
        print(f"  速度: {vel:.3f} rad/s") 
        print(f"  力矩: {applied_torque:.3f} Nm")
        
        print(f"\n📋 结论:")
        if abs(position_value - 1.5) > 0.1 or abs(torque_value) > 0.1:
            print("✅ 滑条控制正常工作!")
            print("🎯 可以继续进行力矩控制测试")
            print("💡 之前的问题是鼠标拖动干扰")
        else:
            print("❌ 滑条仍无响应")
            print("🔧 需要进一步检查代码逻辑")
            
    def close(self):
        """关闭测试"""
        p.disconnect()


def main():
    """主函数"""
    print("🔍 滑条控制调试测试")
    print("=" * 50)
    print("💡 关键修复:")
    print("• 禁用了鼠标拖动干扰")
    print("• 专注测试滑条功能")
    print("• 详细的调试信息输出")
    
    try:
        tester = SliderDebugTest()
        tester.run_slider_debug()
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