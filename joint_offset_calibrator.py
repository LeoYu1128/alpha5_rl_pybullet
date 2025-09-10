#!/usr/bin/env python3
"""
關節零位置校準工具
用於找到每個關節的正確零位置偏移
"""

import pybullet as p
import pybullet_data
import time

class JointOffsetCalibrator:
    def __init__(self):
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        
        self.plane_id = p.loadURDF("plane.urdf")
        
        robot_path = "alpha_description/urdf/alpha_robot_for_pybullet_test.urdf"
        self.robot_id = p.loadURDF(robot_path, [0, 0, 0.4])
        
        self.setup_joints()
        self.create_controls()
        
        # 記錄"正常"位置的偏移
        self.normal_offsets = {}
        
        print("\n=== 關節零位置校準工具 ===")
        print("步驟：")
        print("1. 調整滑桿找到機械臂'正常'姿態")
        print("2. 記錄各關節的'正常'角度")
        print("3. 按's'保存偏移值")
        print("4. 按'g'生成修正後的URDF代碼")
    
    def setup_joints(self):
        self.num_joints = p.getNumJoints(self.robot_id)
        self.controllable_joints = []
        self.joint_names = {}
        
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            if joint_type in [0, 1]:
                self.controllable_joints.append(i)
                self.joint_names[i] = joint_name
    
    def create_controls(self):
        self.sliders = {}
        self.offset_sliders = {}
        
        joint_ranges = [
            (-3.054, 3.054, 0.0),
            (-1.745, 1.745, 1.745),  # joint_2 default to your observation
            (-1.618, 1.618, 0.0),
            (-0.785, 0.785, 0.0),
            (0.001, 0.013, 0.007),
        ]
        
        for i, joint_idx in enumerate(self.controllable_joints):
            if i < len(joint_ranges):
                lower, upper, initial = joint_ranges[i]
            else:
                lower, upper, initial = -3.14, 3.14, 0.0
            
            joint_name = self.joint_names[joint_idx]
            
            # 實際位置滑桿
            slider_id = p.addUserDebugParameter(
                paramName=f"J{joint_idx}_{joint_name}",
                rangeMin=lower,
                rangeMax=upper,
                startValue=initial
            )
            self.sliders[joint_idx] = slider_id
            
            # 偏移量滑桿
            offset_slider_id = p.addUserDebugParameter(
                paramName=f"OFFSET_J{joint_idx}",
                rangeMin=-3.14,
                rangeMax=3.14,
                startValue=0.0
            )
            self.offset_sliders[joint_idx] = offset_slider_id
        
        # 控制按鈕
        self.save_button = p.addUserDebugParameter("保存偏移", 0, 1, 0)
        self.generate_button = p.addUserDebugParameter("生成URDF", 0, 1, 0)
        self.reset_button = p.addUserDebugParameter("重置為零", 0, 1, 0)
        
        self.last_save = 0
        self.last_generate = 0
        self.last_reset = 0
    
    def update_robot(self):
        for joint_idx, slider_id in self.sliders.items():
            target_pos = p.readUserDebugParameter(slider_id)
            
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                maxVelocity=2.0,
                force=1000
            )
    
    def check_buttons(self):
        # 檢查保存按鈕
        current_save = p.readUserDebugParameter(self.save_button)
        if current_save > self.last_save:
            self.save_current_as_normal()
        self.last_save = current_save
        
        # 檢查生成按鈕
        current_generate = p.readUserDebugParameter(self.generate_button)
        if current_generate > self.last_generate:
            self.generate_urdf_corrections()
        self.last_generate = current_generate
        
        # 檢查重置按鈕
        current_reset = p.readUserDebugParameter(self.reset_button)
        if current_reset > self.last_reset:
            self.reset_to_zero()
        self.last_reset = current_reset
    
    def save_current_as_normal(self):
        print("\n=== 保存當前位置為'正常'姿態 ===")
        
        for joint_idx in self.controllable_joints:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            current_pos = joint_state[0]
            self.normal_offsets[joint_idx] = current_pos
            
            joint_name = self.joint_names[joint_idx]
            print(f"  {joint_name}: {current_pos:.4f} 弧度 ({current_pos*180/3.14159:.1f}°)")
        
        print("已保存！現在可以生成URDF修正代碼。")
    
    def generate_urdf_corrections(self):
        if not self.normal_offsets:
            print("請先保存'正常'姿態！")
            return
        
        print("\n=== 生成URDF修正代碼 ===")
        print("將這些修正添加到你的URDF文件中：\n")
        
        joint_names_urdf = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"
        ]
        
        for i, joint_idx in enumerate(self.controllable_joints):
            if joint_idx in self.normal_offsets:
                offset = self.normal_offsets[joint_idx]
                if abs(offset) > 0.001:  # 只顯示有意義的偏移
                    urdf_name = joint_names_urdf[i] if i < len(joint_names_urdf) else f"joint_{joint_idx}"
                    
                    if joint_idx == self.controllable_joints[-1]:  # 棱柱關節
                        print(f"<!-- {urdf_name} - 棱柱關節偏移 -->")
                        print(f'<origin xyz="0 0 {0.009 + offset:.6f}" rpy="0 0 0"/>')
                    else:  # 旋轉關節
                        axis_map = {
                            self.controllable_joints[0]: "0 0",     # Z軸
                            self.controllable_joints[1]: "0",      # Y軸  
                            self.controllable_joints[2]: "0",      # Y軸
                            self.controllable_joints[3]: "0 0"     # Z軸
                        }
                        
                        if joint_idx in axis_map:
                            if "0 0" in axis_map[joint_idx]:  # Z軸旋轉
                                print(f"<!-- {urdf_name} - Z軸旋轉偏移 -->")
                                print(f'<origin xyz="..." rpy="0 0 {offset:.6f}"/>')
                            else:  # Y軸旋轉  
                                print(f"<!-- {urdf_name} - Y軸旋轉偏移 -->")
                                print(f'<origin xyz="..." rpy="0 {offset:.6f} 0"/>')
                    print()
        
        print("複製上述代碼到對應的joint定義中！")
    
    def reset_to_zero(self):
        print("\n重置所有關節到零位置...")
        for joint_idx in self.controllable_joints:
            if joint_idx == self.controllable_joints[-1]:  # 棱柱關節
                target = 0.007
            else:
                target = 0.0
            
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target,
                maxVelocity=1.0,
                force=1000
            )
    
    def run(self):
        print("\n開始校準...")
        
        try:
            while True:
                self.check_buttons()
                self.update_robot()
                
                p.stepSimulation()
                time.sleep(1/240)
                
        except KeyboardInterrupt:
            print("\n校準結束")
        finally:
            p.disconnect()

def main():
    calibrator = JointOffsetCalibrator()
    calibrator.run()

if __name__ == "__main__":
    main()