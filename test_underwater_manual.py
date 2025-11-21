import pybullet as p
import pybullet_data
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
import math
import traceback
import sys

# Import gripper controller
try:
    from envs.test_joint5 import GripperController
    print("Successfully imported GripperController")
except ImportError as e:
    print(f"Failed to import GripperController: {e}")
    print("Make sure test_joint5.py is in the same directory")
    sys.exit(1)

class UnderwaterManualControl:
    """
    Underwater Robotic Arm Manual Control Test Environment
    """
    
    def __init__(self):
        print("Initializing UnderwaterManualControl...")
        
        self.physics_client = None
        self.robot_id = None
        self.target_visual_id = None
        self.plane_id = None
        
        # Gripper controller
        self.gripper = None
        
        # 真实机械臂的home位置（编码器读数）
        self.real_home_positions = [
            np.radians(2.34),
            np.radians(87.8),
            np.radians(1.0),
            np.radians(0.1)
        ]
        
        # URDF中的home位置（PyBullet中的角度）
        self.urdf_home_positions = [
            np.radians(2.34),
            np.radians(87.8),      # URDF中joint_2=0时直立
            np.radians(1.0),
            np.radians(0.1)
        ]
        
        # 角度偏移（真实->URDF需要减去这个值）
        self.angle_offset = np.array([0, np.radians(0.0), 0, 0])
        
        # Gripper home位置
        self.home_gripper_position = 0.0014  # 1.4mm
        
        # Underwater environment parameters
        self.water_density = 1000.0
        self.gravity = -9.81
        self.base_height = 0.1
        self.workspace_radius = 0.4
        
        # Fluid dynamics parameters
        self.drag_coefficient = 0.5
        self.buoyancy_enabled = True
        
        # Water current parameters
        self.current_velocity = np.array([0.1, 0.05, 0.0])
        self.current_variation = True
        self.turbulence_strength = 0.02
        self.time_step = 0
        
        # Joint information
        self.all_joints = {}
        self.main_joint_indices = []
        self.gripper_joint_indices = []
        
        # Control parameters
        self.target_position = np.array([0.3, 0.0, 0.3])
        self.success_threshold = 0.08
        self.current_distance = 0.0
        self.success_achieved = False
        
        # Running state
        self.running = False
        self.simulation_thread = None
        
        # Initialize
        try:
            self._setup_physics()
            self._setup_gui()
            print("Initialization complete")
        except Exception as e:
            print(f"Initialization failed: {e}")
            traceback.print_exc()
            raise
    
    def real_to_urdf(self, real_angles):
        """真实机械臂角度 -> URDF角度（用于控制PyBullet）"""
        return np.array(real_angles) - self.angle_offset
    
    def urdf_to_real(self, urdf_angles):
        """URDF角度 -> 真实机械臂角度（用于显示）"""
        return np.array(urdf_angles) + self.angle_offset
        
    def _setup_physics(self):
        """Setup physics simulation environment"""
        print("\n" + "="*60)
        print("Setting up physics environment...")
        print("="*60)
        
        try:
            # Connect PyBullet
            print("Connecting to PyBullet GUI...")
            self.physics_client = p.connect(p.GUI)
            print(f"Connected (client ID: {self.physics_client})")
            
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # Set underwater gravity
            p.setGravity(0, 0, self.gravity * 0.1)
            p.setTimeStep(1./240.)
            print("Gravity and timestep configured")
            
            # Set camera view
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=30,
                cameraPitch=-20,
                cameraTargetPosition=[0, 0, 0.3]
            )
            
            # Disable mouse picking
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
            print("Camera configured")
            
            # Create seabed floor
            print("Loading ground plane...")
            self.plane_id = p.loadURDF("plane.urdf")
            p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.1, 0.2, 0.4, 1.0])
            print("Ground plane loaded")
            
            # Load Alpha robotic arm
            print("\nLoading robotic arm...")
            self._load_alpha_robot()
            
            # Setup joint mapping
            print("\nSetting up joint mapping...")
            self._setup_joint_mapping()
            
            # Setup underwater physics properties
            print("\nConfiguring underwater physics...")
            self._setup_underwater_physics()
            
            # Add decorations and target
            print("\nAdding decorations...")
            self._add_underwater_decorations()
            
            print("\nCreating target visual...")
            self._create_target_visual()
            
            print("\nPhysics environment setup complete!")
            
        except Exception as e:
            print(f"\nPhysics setup failed: {e}")
            traceback.print_exc()
            raise
        
    def _load_alpha_robot(self):
        """Load Alpha robotic arm"""
        robot_paths = [
            "alpha_description/urdf/alpha_robot_for_pybullet.urdf",
            "alpha_robot_for_pybullet.urdf",
            "../alpha_description/urdf/alpha_robot_for_pybullet.urdf",
        ]
        
        print("Searching for robot URDF...")
        for robot_path in robot_paths:
            print(f"  Trying: {robot_path}")
            if os.path.exists(robot_path):
                try:
                    self.robot_id = p.loadURDF(
                        robot_path, 
                        basePosition=[0, 0, self.base_height], 
                        useFixedBase=True
                    )
                    print(f"  Loaded: {robot_path}")
                    
                    # 立即设置关节到home位置（使用URDF角度）
                    print("  Setting initial home position...")
                    num_joints = p.getNumJoints(self.robot_id)
                    for i in range(num_joints):
                        joint_info = p.getJointInfo(self.robot_id, i)
                        joint_name = joint_info[1].decode('utf-8')
                        
                        if joint_name == 'joint_1':
                            p.resetJointState(self.robot_id, i, self.urdf_home_positions[0])
                        elif joint_name == 'joint_2':
                            p.resetJointState(self.robot_id, i, self.urdf_home_positions[1])
                        elif joint_name == 'joint_3':
                            p.resetJointState(self.robot_id, i, self.urdf_home_positions[2])
                        elif joint_name == 'joint_4':
                            p.resetJointState(self.robot_id, i, self.urdf_home_positions[3])
                        elif joint_name == 'joint_5':
                            p.resetJointState(self.robot_id, i, 0.0014)
                    
                    # 让物理引擎稳定一下
                    for _ in range(100):
                        p.stepSimulation()
                    
                    return
                    
                except Exception as e:
                    print(f"  Loading failed: {e}")
                    continue
            else:
                print(f"  File not found")
        
        # Fallback to KUKA
        print("\nAlpha robot not found, trying KUKA fallback...")
        try:
            self.robot_id = p.loadURDF(
                "kuka_iiwa/model.urdf",
                basePosition=[0, 0, self.base_height],
                useFixedBase=True
            )
            print("Loaded KUKA as fallback")
        except Exception as e:
            print(f"KUKA loading also failed: {e}")
            raise FileNotFoundError("Cannot find any robotic arm URDF file")
    
    def _setup_joint_mapping(self):
        """Setup joint mapping"""
        num_joints = p.getNumJoints(self.robot_id)
        print(f"Robot has {num_joints} joints")
        
        self.all_joints = {}
        self.main_joint_indices = []
        self.gripper_joint_indices = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            max_velocity = joint_info[11]
            
            self.all_joints[i] = {
                'name': joint_name,
                'type': joint_type,
                'lower': lower_limit,
                'upper': upper_limit,
                'max_velocity': max_velocity
            }
            
            print(f"  Joint {i}: {joint_name} (type={joint_type}, "
                  f"range=[{lower_limit:.3f}, {upper_limit:.3f}], "
                  f"max_vel={max_velocity:.3f})")
            
            # Identify main control joints
            if joint_name in ['joint_1', 'joint_2', 'joint_3', 'joint_4']:
                self.main_joint_indices.append(i)
                print(f"    -> Main joint")
            elif 'jaw' in joint_name.lower() or joint_name == 'joint_5':
                self.gripper_joint_indices.append(i)
                print(f"    -> Gripper joint")
        
        if len(self.main_joint_indices) != 4:
            print(f"Warning: Expected 4 main joints, found {len(self.main_joint_indices)}")
        
        print(f"\nMain joints: {self.main_joint_indices}")
        print(f"Gripper joints: {self.gripper_joint_indices}")
        
        # Find end effector
        self.tcp_index = None
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if 'tcp' in joint_name.lower():
                self.tcp_index = i
                break
        
        if self.tcp_index is None:
            self.tcp_index = num_joints - 1
        
        print(f"TCP index: {self.tcp_index}")
        
        # Initialize gripper controller
        print("\nInitializing gripper controller...")
        try:
            self.gripper = GripperController(self.robot_id)
            print("Gripper controller initialized")
        except Exception as e:
            print(f"Gripper controller failed: {e}")
            print("Continuing without gripper control...")
            self.gripper = None
    
    def _setup_underwater_physics(self):
        """Setup underwater physics properties"""
        num_joints = p.getNumJoints(self.robot_id)
        
        for i in range(-1, num_joints):
            if i >= 0:
                p.changeDynamics(
                    self.robot_id, i,
                    linearDamping=2.0,
                    angularDamping=2.0,
                    jointDamping=0.5
                )
            else:
                p.changeDynamics(
                    self.robot_id, i,
                    linearDamping=1.0,
                    angularDamping=1.0
                )
            
            if self.buoyancy_enabled:
                mass_info = p.getDynamicsInfo(self.robot_id, i)
                original_mass = mass_info[0]
                
                if original_mass > 0:
                    buoyancy_factor = 0.7
                    effective_mass = original_mass * buoyancy_factor
                    p.changeDynamics(self.robot_id, i, mass=effective_mass)
        
        print("Underwater physics configured")
                    
    def _add_underwater_decorations(self):
        """Add underwater decorations"""
        try:
            for i in range(3):
                x = np.random.uniform(-0.8, 0.8)
                y = np.random.uniform(-0.8, 0.8) 
                z = np.random.uniform(0.1, 0.5)
                
                visual_shape = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=0.05,
                    rgbaColor=[0.1, 0.4, 0.2, 0.8]
                )
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x, y, z]
                )
            print("Decorations added")
        except Exception as e:
            print(f"Decoration warning: {e}")
            
    def _create_target_visual(self):
        """Create target position visualization"""
        if self.target_visual_id is not None:
            try:
                p.removeBody(self.target_visual_id)
            except:
                pass
        
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.04,
            rgbaColor=[1, 0.5, 0, 0.9]
        )
        
        self.target_visual_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_position
        )
        print("Target visual created")
    
    def _update_water_current(self):
        """Update water current velocity"""
        if self.current_variation:
            time_factor = self.time_step * 0.01
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
        """Apply underwater forces"""
        self._update_water_current()
        try:
            ee_state = p.getLinkState(self.robot_id, self.tcp_index, computeLinkVelocity=1)
            ee_velocity = np.array(ee_state[6])
            relative_velocity = ee_velocity - self.current_velocity_actual
            drag_force = -0.5 * self.water_density * self.drag_coefficient * 0.01 * \
                         relative_velocity * np.linalg.norm(relative_velocity)
            max_force = 10.0
            drag_force = np.clip(drag_force, -max_force, max_force)
            p.applyExternalForce(self.robot_id, self.tcp_index, forceObj=drag_force,
                               posObj=[0, 0, 0], flags=p.LINK_FRAME)
        except:
            pass
    
    def get_main_joint_positions(self):
        """Get current positions of main joints"""
        positions = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            positions.append(joint_state[0])
        return np.array(positions)
    
    def apply_position_control(self, target_positions):
        """Apply position control with joint limits and velocity limits"""
        if len(target_positions) != 4:
            return
        
        # Apply joint limits
        for i, joint_idx in enumerate(self.main_joint_indices):
            if joint_idx in self.all_joints:
                joint = self.all_joints[joint_idx]
                target_positions[i] = np.clip(
                    target_positions[i], 
                    joint['lower'], 
                    joint['upper']
                )
        
        # Apply position control with joint max velocities
        for i, joint_idx in enumerate(self.main_joint_indices):
            if joint_idx in self.all_joints:
                joint = self.all_joints[joint_idx]
                max_vel = joint.get('max_velocity', 1.0)
                
                p.setJointMotorControl2(
                    self.robot_id, 
                    joint_idx, 
                    p.POSITION_CONTROL,
                    targetPosition=target_positions[i], 
                    maxVelocity=max_vel,
                    force=9.0
                )
            
    def _get_end_effector_position(self):
        """Get end effector position"""
        try:
            link_state = p.getLinkState(self.robot_id, self.tcp_index)
            return np.array(link_state[0])
        except:
            return np.array([0, 0, 0])
            
    def _check_success(self):
        """Check if target is reached"""
        ee_pos = self._get_end_effector_position()
        self.current_distance = np.linalg.norm(ee_pos - self.target_position)
        
        if self.current_distance < self.success_threshold:
            if not self.success_achieved:
                self.success_achieved = True
                self._show_success_message()
        else:
            self.success_achieved = False
            
    def _show_success_message(self):
        """Show success message"""
        def show_msg():
            messagebox.showinfo("Success!", f"Target reached!\nDistance: {self.current_distance:.3f}m")
        self.root.after(0, show_msg)
        
    def _setup_gui(self):
        """Setup GUI"""
        print("\n" + "="*60)
        print("Setting up GUI...")
        print("="*60)
        
        self.root = tk.Tk()
        self.root.title("Underwater Alpha Robotic Arm (with Gripper)")
        self.root.geometry("450x900")
        
        # Title
        title_label = tk.Label(self.root, text="Underwater Arm Control", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Joint control
        control_frame = ttk.LabelFrame(self.root, text="Joint Control", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        self.joint_vars = []
        self.joint_scales = []
        
        # 关节配置（name, lower, upper, real_home_deg）
        joint_configs = [
            ("Base (joint_1)", 0, 5.725, 2.34),
            ("Shoulder (joint_2)", 0, 3.0, 87.8),
            ("Elbow (joint_3)", 0, 3.228, 1.0),
            ("Wrist (joint_4)", 0, 5.725, 0.1)
        ]
        
        for i, (name, lower, upper, real_home_deg) in enumerate(joint_configs):
            joint_frame = ttk.Frame(control_frame)
            joint_frame.pack(fill="x", pady=5)
            
            label = ttk.Label(joint_frame, text=f"{name}:")
            label.pack(side="left", anchor="w")
            
            # GUI滑块使用URDF角度
            var = tk.DoubleVar(value=self.urdf_home_positions[i])
            self.joint_vars.append(var)
            
            scale = ttk.Scale(joint_frame, 
                            from_=lower, 
                            to=upper, 
                            variable=var, 
                            orient="horizontal",
                            command=self._on_joint_change)
            scale.pack(side="left", fill="x", expand=True, padx=5)
            self.joint_scales.append(scale)
            
            # 显示真实角度
            real_angle = self.urdf_to_real([self.urdf_home_positions[i]])[0]
            value_label = ttk.Label(joint_frame, text=f"{np.degrees(real_angle):.1f}°", width=10)
            value_label.pack(side="right")
            
            def make_update_func(lbl, v, idx):
                def update(*args):
                    urdf_val = v.get()
                    real_val = urdf_val + self.angle_offset[idx]
                    lbl.config(text=f"{np.degrees(real_val):.1f}°")
                return update
            
            var.trace('w', make_update_func(value_label, var, i))
        
        # Gripper control
        gripper_frame = ttk.LabelFrame(self.root, text="Gripper Control", padding=10)
        gripper_frame.pack(fill="x", padx=10, pady=5)
        
        gripper_control_frame = ttk.Frame(gripper_frame)
        gripper_control_frame.pack(fill="x", pady=5)
        
        label = ttk.Label(gripper_control_frame, text="Gripper:")
        label.pack(side="left", anchor="w")
        
        gripper_home_normalized = (self.home_gripper_position - 0.00137) / (0.0133 - 0.00137)
        self.gripper_var = tk.DoubleVar(value=gripper_home_normalized)
        
        gripper_scale = ttk.Scale(gripper_control_frame, from_=0.0, to=1.0,
                                 variable=self.gripper_var, orient="horizontal",
                                 command=self._on_gripper_change)
        gripper_scale.pack(side="left", fill="x", expand=True, padx=5)
        
        gripper_value_label = ttk.Label(gripper_control_frame, text="1.40mm", width=14)
        gripper_value_label.pack(side="right")
        
        def update_gripper_label(*args):
            val = self.gripper_var.get()
            actual_distance = 0.00137 + val * (0.0133 - 0.00137)
            actual_mm = actual_distance * 1000
            status = "(Closed)" if val < 0.3 else "(Open)" if val > 0.7 else "(Half)"
            gripper_value_label.config(text=f"{actual_mm:.2f}mm {status}")
        
        self.gripper_var.trace('w', update_gripper_label)
        
        button_frame = ttk.Frame(gripper_frame)
        button_frame.pack(fill="x", pady=5)
        ttk.Button(button_frame, text="Open", command=lambda: self.gripper_var.set(1.0)).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Close", command=lambda: self.gripper_var.set(0.0)).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Home (1.4mm)", command=lambda: self.gripper_var.set(gripper_home_normalized)).pack(side="left", padx=5)
        
        # Target position control
        target_frame = ttk.LabelFrame(self.root, text="Target Position", padding=10)
        target_frame.pack(fill="x", padx=10, pady=5)
        
        self.target_vars = []
        for i, label_text in enumerate(["X:", "Y:", "Z:"]):
            target_row = ttk.Frame(target_frame)
            target_row.pack(fill="x", pady=2)
            
            ttk.Label(target_row, text=label_text).pack(side="left")
            
            var = tk.DoubleVar(value=self.target_position[i])
            self.target_vars.append(var)
            
            scale = ttk.Scale(target_row, from_=-0.5, to=0.8, variable=var, orient="horizontal",
                            command=self._on_target_change)
            scale.pack(side="left", fill="x", expand=True, padx=5)
            
            value_label = ttk.Label(target_row, text=f"{self.target_position[i]:.2f}")
            value_label.pack(side="right")
            var.trace('w', lambda *args, lbl=value_label, v=var: lbl.config(text=f"{v.get():.2f}"))
        
        # Water parameters
        water_frame = ttk.LabelFrame(self.root, text="Water Parameters", padding=10)
        water_frame.pack(fill="x", padx=10, pady=5)
        
        self.water_var = tk.BooleanVar(value=self.current_variation)
        ttk.Checkbutton(water_frame, text="Dynamic Water Current", 
                       variable=self.water_var, command=self._on_water_toggle).pack()
        
        turbulence_row = ttk.Frame(water_frame)
        turbulence_row.pack(fill="x", pady=2)
        ttk.Label(turbulence_row, text="Turbulence:").pack(side="left")
        self.turbulence_var = tk.DoubleVar(value=self.turbulence_strength)
        ttk.Scale(turbulence_row, from_=0, to=0.1, variable=self.turbulence_var, 
                 orient="horizontal", command=self._on_turbulence_change).pack(side="left", fill="x", expand=True, padx=5)
        
        # Status
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        self.distance_label = ttk.Label(status_frame, text="Distance: --")
        self.distance_label.pack()
        
        self.success_label = ttk.Label(status_frame, text="Status: Not reached", foreground="red")
        self.success_label.pack()
        
        self.water_label = ttk.Label(status_frame, text="Water speed: --")
        self.water_label.pack()
        
        self.joint_pos_label = ttk.Label(status_frame, text="Joints: --", font=("Courier", 8))
        self.joint_pos_label.pack()
        
        self.gripper_status_label = ttk.Label(status_frame, text="Gripper: --")
        self.gripper_status_label.pack()
        
        # Control buttons
        button_control_frame = ttk.Frame(self.root)
        button_control_frame.pack(fill="x", padx=10, pady=10)
        
        self.start_button = ttk.Button(button_control_frame, text="Start", command=self._start_simulation)
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(button_control_frame, text="Stop", command=self._stop_simulation)
        self.stop_button.pack(side="left", padx=5)
        
        ttk.Button(button_control_frame, text="Reset Target", command=self._reset_target).pack(side="left", padx=5)
        ttk.Button(button_control_frame, text="Home Position", command=self._reset_to_home).pack(side="left", padx=5)
        
        # Info panel
        info_frame = ttk.LabelFrame(self.root, text="Robot Configuration", padding=5)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        info_text = (
            "Real Home Position:\n"
            "Base:2.34° | Shoulder:87.8° | Elbow:1° | Wrist:0.1° | Gripper:1.4mm\n\n"
            "Joint Ranges:\n"
            "Base:0-328° | Shoulder:±100° | Elbow:0-185° | Wrist:0-328°\n\n"
            "Note: Display shows real angles, internal uses URDF angles\n"
            "(joint_2: real=87.8° when URDF=0°)"
        )
        ttk.Label(info_frame, text=info_text, font=("Courier", 8), justify="left").pack()
        
        print("GUI setup complete")
        
    def _on_joint_change(self, *args):
        """Handle joint slider changes"""
        if not self.running:
            return
        # GUI滑块的值是URDF角度，直接使用
        target_positions = [var.get() for var in self.joint_vars]
        self.apply_position_control(target_positions)
    
    def _on_gripper_change(self, *args):
        """Handle gripper slider changes"""
        if not self.running or self.gripper is None:
            return
        self.gripper.control(self.gripper_var.get())
                
    def _on_target_change(self, *args):
        """Handle target position changes"""
        self.target_position = np.array([var.get() for var in self.target_vars])
        self._create_target_visual()
        
    def _on_water_toggle(self):
        """Handle water current toggle"""
        self.current_variation = self.water_var.get()
        
    def _on_turbulence_change(self, *args):
        """Handle turbulence slider changes"""
        self.turbulence_strength = self.turbulence_var.get()
        
    def _reset_target(self):
        """Reset target to random position"""
        r = np.random.uniform(0.15, self.workspace_radius)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi)* np.sin(theta)
        z = np.clip(r * np.cos(phi) + self.base_height, 0.1, 0.7)
        
        self.target_position = np.array([x, y, z])
        for i, var in enumerate(self.target_vars):
            var.set(self.target_position[i])
        self._create_target_visual()
        self.success_achieved = False
        
    def _reset_to_home(self):
        """Reset to home position"""
        print("\nResetting to home position...")
        
        # 使用URDF角度重置关节
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.resetJointState(self.robot_id, joint_idx, self.urdf_home_positions[i])
            real_angle = self.real_home_positions[i]
            urdf_angle = self.urdf_home_positions[i]
            print(f"  Joint {i+1}: Real={np.degrees(real_angle):.2f}°, "
                  f"URDF={np.degrees(urdf_angle):.2f}°")
        
        # 更新GUI滑块（使用URDF角度）
        for i, var in enumerate(self.joint_vars):
            var.set(self.urdf_home_positions[i])
        
        # Reset gripper
        gripper_home_normalized = (self.home_gripper_position - 0.00137) / (0.0133 - 0.00137)
        self.gripper_var.set(gripper_home_normalized)
        if self.gripper is not None:
            self.gripper.control(gripper_home_normalized)
        print(f"  Gripper: {self.home_gripper_position*1000:.2f}mm")
        
        self.success_achieved = False
        print("Reset complete!")
        
    def _start_simulation(self):
        """Start simulation"""
        if not self.running:
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.simulation_thread = threading.Thread(target=self._simulation_loop)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
    def _stop_simulation(self):
        """Stop simulation"""
        self.running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        
    def _simulation_loop(self):
        """Main simulation loop"""
        while self.running:
            try:
                self._apply_underwater_forces()
                p.stepSimulation()
                self._check_success()
                self._update_status_display()
                self.time_step += 1
                time.sleep(1/240.0)
            except Exception as e:
                print(f"Simulation error: {e}")
                break
                
    def _update_status_display(self):
        """Update status display in GUI"""
        def update():
            # Distance to target
            self.distance_label.config(text=f"Distance: {self.current_distance:.3f}m")
            
            # Success status
            if self.success_achieved:
                self.success_label.config(text="Status: Reached!", foreground="green")
            else:
                self.success_label.config(text="Status: Not reached", foreground="red")
            
            # Water current speed
            if hasattr(self, 'current_velocity_actual'):
                water_speed = np.linalg.norm(self.current_velocity_actual)
                self.water_label.config(text=f"Water: {water_speed:.3f}m/s")
            
            # Joint positions - 显示真实角度
            joint_pos_urdf = self.get_main_joint_positions()
            joint_pos_real = self.urdf_to_real(joint_pos_urdf)
            joint_degrees = [f'{np.degrees(p):.1f}' for p in joint_pos_real]
            self.joint_pos_label.config(text=f"Real Angles: [{', '.join(joint_degrees)}]°")
            
            # Gripper status
            if self.gripper is not None:
                try:
                    gripper_state = self.gripper.get_state()
                    val = gripper_state['normalized']
                    actual_distance = 0.00137 + val * (0.0133 - 0.00137)
                    actual_mm = actual_distance * 1000
                    status = "Closed" if val < 0.3 else "Open" if val > 0.7 else "Half"
                    self.gripper_status_label.config(text=f"Gripper: {actual_mm:.2f}mm ({status})")
                except:
                    self.gripper_status_label.config(text="Gripper: N/A")
            else:
                self.gripper_status_label.config(text="Gripper: N/A")
                
        self.root.after(0, update)
        
    def run(self):
        """Run the application"""
        print("\n" + "="*60)
        print("Starting GUI...")
        print("="*60)
        print("Use sliders to control joints and gripper!")
        print("Try to reach the orange target sphere!")
        print("\nReal Home Position:")
        for i, pos in enumerate(self.real_home_positions):
            print(f"  Joint {i+1}: {np.degrees(pos):.2f}° (real)")
        print(f"\nURDF Home Position (internal):")
        for i, pos in enumerate(self.urdf_home_positions):
            print(f"  Joint {i+1}: {np.degrees(pos):.2f}° (URDF)")
        print(f"\nAngle Offset (Real - URDF):")
        for i, offset in enumerate(self.angle_offset):
            print(f"  Joint {i+1}: {np.degrees(offset):.2f}°")
        print("\n")
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nUser interrupted")
        finally:
            self._cleanup()
            
    def _cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
        print("Cleanup complete")


if __name__ == "__main__":
    print("="*60)
    print("Underwater Alpha Arm Manual Control (with Gripper)")
    print("="*60)
    print("\nRobot Configuration:")
    print("\nReal Home Position (Physical Zero):")
    print("  Base (joint_1):     2.34°")
    print("  Shoulder (joint_2): 87.8°")
    print("  Elbow (joint_3):    1°")
    print("  Wrist (joint_4):    0.1°")
    print("  Gripper:            1.4mm")
    print("\nURDF Home Position (PyBullet Internal):")
    print("  Base (joint_1):     2.34°")
    print("  Shoulder (joint_2): 0° (offset by 87.8°)")
    print("  Elbow (joint_3):    1°")
    print("  Wrist (joint_4):    0.1°")
    print("\nNote: GUI displays real angles, but controls URDF angles")
    print("="*60)
    
    try:
        app = UnderwaterManualControl()
        app.run()
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")