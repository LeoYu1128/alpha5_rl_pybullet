#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")
robot_id = p.loadURDF("alpha_description/urdf/alpha_robot_for_pybullet.urdf", [0, 0, 0.5])

print("PyBullet不支持URDF mimic標籤，手動創建mimic約束...")

# 先禁用所有夾爪關節的馬達
p.setJointMotorControl2(robot_id, 9, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
p.setJointMotorControl2(robot_id, 10, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

# 創建GEAR約束來實現mimic功能
# joint_5 (index 7) 控制 left gripper (index 9)
gear1 = p.createConstraint(robot_id, 7, robot_id, 9, 
                          jointType=p.JOINT_GEAR,
                          jointAxis=[0, 0, 1],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
p.changeConstraint(gear1, gearRatio=51, maxForce=10000)

# joint_5 (index 7) 控制 right gripper (index 10) - 相反方向
gear2 = p.createConstraint(robot_id, 7, robot_id, 10,
                          jointType=p.JOINT_GEAR, 
                          jointAxis=[0, 0, 1],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
p.changeConstraint(gear2, gearRatio=-51, maxForce=10000)  # 負號表示相反方向

# 創建滑桿控制joint_5
joint_5_slider = p.addUserDebugParameter("joint_5 (推桿)", 0.0013, 0.0133, 0.007)

print("現在joint_5應該能控制夾爪開合了!")
print("滑桿向右 = 夾爪張開")

try:
    while True:
        # 只控制joint_5，夾爪會通過gear約束自動跟隨
        j5_val = p.readUserDebugParameter(joint_5_slider)
        p.setJointMotorControl2(robot_id, 7, p.POSITION_CONTROL, targetPosition=j5_val, force=1000)
        
        # 每秒打印狀態
        if int(time.time() * 2) % 2 == 0:
            j5_actual = p.getJointState(robot_id, 7)[0]
            jaw1_actual = p.getJointState(robot_id, 9)[0]
            jaw2_actual = p.getJointState(robot_id, 10)[0]
            print(f"推桿: {j5_actual:.4f} | 左夾爪: {jaw1_actual:.4f} | 右夾爪: {jaw2_actual:.4f}")
        
        p.stepSimulation()
        time.sleep(1/240)
        
except KeyboardInterrupt:
    print("測試結束 - 如果看到夾爪跟著推桿動，說明解決方案成功!")
    p.disconnect()