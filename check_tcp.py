import pybullet as p
import pybullet_data
import time

# 连接
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# 加载机械臂
robot = p.loadURDF("alpha_description/urdf/alpha_robot_for_pybullet.urdf", 
                   basePosition=[0, 0, 0.1], 
                   useFixedBase=True)

# 找到TCP
num_joints = p.getNumJoints(robot)
tcp_index = None

for i in range(num_joints):
    joint_info = p.getJointInfo(robot, i)
    joint_name = joint_info[1].decode('utf-8')
    link_name = joint_info[12].decode('utf-8')
    
    print(f"Joint {i}: {joint_name} -> Link: {link_name}")
    
    if 'tcp' in link_name.lower():
        tcp_index = i
        print(f"  ⭐ 找到TCP！索引={i}")

# 获取TCP位置
if tcp_index is not None:
    link_state = p.getLinkState(robot, tcp_index)
    tcp_pos = link_state[0] # 世界坐标系中的位置
    print(f"\n✅ TCP位置: {tcp_pos}")
    print(f"   距离地面: {tcp_pos[2]:.3f}m")
    
    # 可视化TCP位置（画一个红色小球）
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, 
                                 rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, 
                     basePosition=tcp_pos)
    
    print("\n🔴 红色小球标记了TCP位置")

# 也检查夹爪finger的位置
for i in range(num_joints):
    joint_info = p.getJointInfo(robot, i)
    link_name = joint_info[12].decode('utf-8')
    
    if 'rs1_130' in link_name.lower() or 'rs1_139' in link_name.lower():
        link_state = p.getLinkState(robot, i)
        finger_pos = link_state[0]
        print(f"\nFinger {link_name}: {finger_pos}")

# 保持窗口打开
while True:
    p.stepSimulation()
    time.sleep(1./240.)