import pybullet as p
import pybullet_data
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
import math

class UnderwaterManualControl:
    """
    Underwater Robotic Arm Manual Control Test Environment
    Based on AlphaRobotController joint control logic
    """
    
    def __init__(self):
        self.physics_client = None
        self.robot_id = None
        self.target_visual_id = None
        self.plane_id = None
        
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
        
        # Joint information - based on your code structure
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
        
        # Initialize physics environment and GUI
        self._setup_physics()
        self._setup_gui()
        
    def _setup_physics(self):
        """Setup physics simulation environment - based on your code"""
        print("Initializing underwater physics environment...")
        
        # Connect PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set underwater gravity
        p.setGravity(0, 0, self.gravity * 0.1)  # Reduced gravity underwater
        p.setTimeStep(1./240.)
        
        # Set camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=30,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.3]
        )
        
        # Disable mouse picking
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        
        # Create seabed floor
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.1, 0.2, 0.4, 1.0])
        
        # Load Alpha robotic arm - using your loading logic
        self._load_alpha_robot()
        
        # Setup joint mapping - using your mapping logic
        self._setup_joint_mapping()
        
        # Setup underwater physics properties
        self._setup_underwater_physics()
        
        # Add decorations and target
        self._add_underwater_decorations()
        self._create_target_visual()
        
        print("Underwater environment initialization completed!")
        
    def _load_alpha_robot(self):
        """Load Alpha robotic arm - copy your logic"""
        robot_paths = [
            "alpha_robot_for_pybullet.urdf",
            "alpha_description/urdf/alpha_robot_for_pybullet.urdf",
            "../alpha_description/urdf/alpha_robot_for_pybullet.urdf"
        ]
        
        self.robot_id = None
        for robot_path in robot_paths:
            if os.path.exists(robot_path):
                try:
                    self.robot_id = p.loadURDF(
                        robot_path, 
                        basePosition=[0, 0, self.base_height], 
                        useFixedBase=True
                    )
                    print(f"Successfully loaded Alpha robotic arm: {robot_path}")
                    return
                except Exception as e:
                    print(f"Loading failed: {e}")
                    continue
        
        # If Alpha arm not found, use KUKA as fallback
        try:
            self.robot_id = p.loadURDF(
                "kuka_iiwa/model.urdf",
                basePosition=[0, 0, self.base_height],
                useFixedBase=True
            )
            print("Using KUKA robotic arm as fallback")
        except:
            raise FileNotFoundError("Cannot find any robotic arm URDF file")
    
    def _setup_joint_mapping(self):
        """Setup joint mapping - copy your logic"""
        num_joints = p.getNumJoints(self.robot_id)
        
        # Store all joint information
        self.all_joints = {}
        self.main_joint_indices = []
        self.gripper_joint_indices = []
        
        print(f"Analyzing robotic arm joint structure, total joints: {num_joints}")
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            self.all_joints[i] = {
                'name': joint_name,
                'type': joint_type,
                'lower': lower_limit,
                'upper': upper_limit
            }
            
            print(f"Joint {i}: {joint_name}, type: {joint_type}, range: [{lower_limit:.2f}, {upper_limit:.2f}]")
            
            # Classify joints - using your logic
            if joint_name in ['joint_1', 'joint_2', 'joint_3', 'joint_4']:
                self.main_joint_indices.append(i)
            elif 'jaw' in joint_name.lower() or joint_name == 'joint_5':
                self.gripper_joint_indices.append(i)
            elif joint_type == p.JOINT_REVOLUTE and lower_limit < upper_limit:
                # For KUKA and other arms, select controllable revolute joints
                if len(self.main_joint_indices) < 4:
                    self.main_joint_indices.append(i)
        
        # Ensure at least 4 main joints
        if len(self.main_joint_indices) < 4:
            # Add first few joints
            for i in range(min(4, num_joints)):
                if i not in self.main_joint_indices:
                    self.main_joint_indices.append(i)
        
        # Take only first 4
        self.main_joint_indices = self.main_joint_indices[:4]
        
        print(f"Main control joints: {self.main_joint_indices}")
        print(f"Gripper joints: {self.gripper_joint_indices}")
        
        # Find end effector
        self.tcp_index = num_joints - 1 if num_joints > 0 else 0
        
    def _setup_underwater_physics(self):
        """Setup underwater physics properties"""
        print("Setting up underwater physics properties...")
        
        num_joints = p.getNumJoints(self.robot_id)
        
        # Set underwater properties for each link
        for i in range(-1, num_joints):
            # Increase damping to simulate water resistance
            if i >= 0:
                p.changeDynamics(
                    self.robot_id, i,
                    linearDamping=2.0,      # Linear damping
                    angularDamping=2.0,     # Angular damping
                    jointDamping=0.5        # Joint damping
                )
            else:
                # Base link
                p.changeDynamics(
                    self.robot_id, i,
                    linearDamping=1.0,
                    angularDamping=1.0
                )
            
            # Simulate buoyancy effect
            if self.buoyancy_enabled:
                mass_info = p.getDynamicsInfo(self.robot_id, i)
                original_mass = mass_info[0]
                
                if original_mass > 0:
                    # Buoyancy counteracts 70% of weight
                    buoyancy_factor = 0.7
                    effective_mass = original_mass * buoyancy_factor
                    
                    p.changeDynamics(
                        self.robot_id, i,
                        mass=effective_mass
                    )
                    
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
                decoration_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[x, y, z]
                )
            
            print("Added underwater decorations")
            
        except Exception as e:
            print(f"Failed to add decorations: {e}")
            
    def _create_target_visual(self):
        """Create target position visualization"""
        if self.target_visual_id is not None:
            try:
                p.removeBody(self.target_visual_id)
            except:
                pass
        
        # Create orange sphere as target
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.04,
            rgbaColor=[1, 0.5, 0, 0.9]  # Orange
        )
        
        self.target_visual_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_position
        )
        
    def _update_water_current(self):
        """Update water current velocity"""
        if self.current_variation:
            time_factor = self.time_step * 0.01
            
            # Base current + periodic variation + turbulence
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
        """Apply underwater fluid forces"""
        self._update_water_current()
        
        # Get end effector state
        try:
            ee_state = p.getLinkState(
                self.robot_id, self.tcp_index,
                computeLinkVelocity=1
            )
            ee_velocity = np.array(ee_state[6])  # Linear velocity
            
            # Calculate velocity relative to water current
            relative_velocity = ee_velocity - self.current_velocity_actual
            
            # Calculate fluid drag force
            drag_force = -0.5 * self.water_density * self.drag_coefficient * 0.01 * \
                         relative_velocity * np.linalg.norm(relative_velocity)
            
            # Limit force magnitude
            max_force = 10.0
            drag_force = np.clip(drag_force, -max_force, max_force)
            
            # Apply drag force
            p.applyExternalForce(
                self.robot_id, self.tcp_index,
                forceObj=drag_force,
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME
            )
            
        except Exception as e:
            # Skip force application if getting link state fails
            pass
    
    def get_main_joint_positions(self):
        """Get current positions of main joints - copy your logic"""
        positions = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            positions.append(joint_state[0])
        return np.array(positions)
    
    def apply_position_control(self, target_positions):
        """
        Apply position control - based on your logic, but directly set absolute positions
        """
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
        
        # Execute position control - using your parameters
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                maxVelocity=1.0,  # Slightly slower underwater
                force=500         # Higher force needed underwater to overcome resistance
            )
            
    def _get_end_effector_position(self):
        """Get end effector position"""
        try:
            link_state = p.getLinkState(self.robot_id, self.tcp_index)
            return np.array(link_state[0])
        except:
            return np.array([0, 0, 0])
            
    def _check_success(self):
        """Check if successfully reached target"""
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
            messagebox.showinfo("Success!", f"Robotic arm successfully reached target position!\nDistance: {self.current_distance:.3f}m")
        
        # Show message in main thread
        self.root.after(0, show_msg)
        
    def _setup_gui(self):
        """Setup GUI interface"""
        self.root = tk.Tk()
        self.root.title("Underwater Alpha Robotic Arm Manual Control")
        self.root.geometry("450x700")
        
        # Title
        title_label = tk.Label(self.root, text="Underwater Robotic Arm Control Panel", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(self.root, text="Joint Control", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        self.joint_vars = []
        self.joint_scales = []
        
        # Create sliders for 4 main joints
        joint_names = ["Joint 1 (Base)", "Joint 2 (Shoulder)", "Joint 3 (Elbow)", "Joint 4 (Wrist)"]
        for i in range(4):
            joint_frame = ttk.Frame(control_frame)
            joint_frame.pack(fill="x", pady=5)
            
            # Joint label
            label = ttk.Label(joint_frame, text=f"{joint_names[i]}:")
            label.pack(side="left", anchor="w")
            
            # Joint variable
            var = tk.DoubleVar(value=0.0)
            self.joint_vars.append(var)
            
            # Slider - default range, will be updated based on actual joints
            scale = ttk.Scale(joint_frame, from_=-3.14, to=3.14, 
                            variable=var, orient="horizontal",
                            command=self._on_joint_change)
            scale.pack(side="left", fill="x", expand=True, padx=5)
            self.joint_scales.append(scale)
            
            # Value display
            value_label = ttk.Label(joint_frame, text="0.00", width=8)
            value_label.pack(side="right")
            var.trace('w', lambda *args, lbl=value_label, v=var: 
                     lbl.config(text=f"{v.get():.2f}"))
        
        # Target position control
        target_frame = ttk.LabelFrame(self.root, text="Target Position", padding=10)
        target_frame.pack(fill="x", padx=10, pady=5)
        
        self.target_vars = []
        target_labels = ["X:", "Y:", "Z:"]
        
        for i, label_text in enumerate(target_labels):
            target_row = ttk.Frame(target_frame)
            target_row.pack(fill="x", pady=2)
            
            label = ttk.Label(target_row, text=label_text)
            label.pack(side="left")
            
            var = tk.DoubleVar(value=self.target_position[i])
            self.target_vars.append(var)
            
            scale = ttk.Scale(target_row, from_=-0.5, to=0.8, 
                            variable=var, orient="horizontal",
                            command=self._on_target_change)
            scale.pack(side="left", fill="x", expand=True, padx=5)
            
            value_label = ttk.Label(target_row, text=f"{self.target_position[i]:.2f}")
            value_label.pack(side="right")
            var.trace('w', lambda *args, lbl=value_label, v=var:
                     lbl.config(text=f"{v.get():.2f}"))
        
        # Water current parameter control
        water_frame = ttk.LabelFrame(self.root, text="Water Current Parameters", padding=10)
        water_frame.pack(fill="x", padx=10, pady=5)
        
        # Water current toggle
        self.water_var = tk.BooleanVar(value=self.current_variation)
        water_check = ttk.Checkbutton(water_frame, text="Enable Dynamic Water Current", 
                                    variable=self.water_var,
                                    command=self._on_water_toggle)
        water_check.pack()
        
        # Turbulence intensity
        turbulence_row = ttk.Frame(water_frame)
        turbulence_row.pack(fill="x", pady=2)
        
        ttk.Label(turbulence_row, text="Turbulence Intensity:").pack(side="left")
        self.turbulence_var = tk.DoubleVar(value=self.turbulence_strength)
        turbulence_scale = ttk.Scale(turbulence_row, from_=0, to=0.1,
                                   variable=self.turbulence_var, orient="horizontal",
                                   command=self._on_turbulence_change)
        turbulence_scale.pack(side="left", fill="x", expand=True, padx=5)
        
        # Status display
        status_frame = ttk.LabelFrame(self.root, text="Status Information", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        self.distance_label = ttk.Label(status_frame, text="Distance to target: --")
        self.distance_label.pack()
        
        self.success_label = ttk.Label(status_frame, text="Status: Not reached", 
                                     foreground="red")
        self.success_label.pack()
        
        self.water_label = ttk.Label(status_frame, text="Water current speed: --")
        self.water_label.pack()
        
        self.joint_pos_label = ttk.Label(status_frame, text="Joint positions: --", 
                                        font=("Courier", 8))
        self.joint_pos_label.pack()
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Simulation", 
                                      command=self._start_simulation)
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Simulation", 
                                     command=self._stop_simulation)
        self.stop_button.pack(side="left", padx=5)
        
        reset_button = ttk.Button(button_frame, text="Reset Target", 
                                 command=self._reset_target)
        reset_button.pack(side="left", padx=5)
        
        home_button = ttk.Button(button_frame, text="Return to Home Position",
                                command=self._reset_to_home)
        home_button.pack(side="left", padx=5)
        
        # Update joint slider ranges
        self._update_joint_ranges()
        
    def _update_joint_ranges(self):
        """Update slider ranges based on actual joints"""
        for i, joint_idx in enumerate(self.main_joint_indices):
            if i < len(self.joint_scales) and joint_idx in self.all_joints:
                joint = self.all_joints[joint_idx]
                lower = joint['lower'] if joint['lower'] > -100 else -3.14
                upper = joint['upper'] if joint['upper'] < 100 else 3.14
                
                self.joint_scales[i].config(from_=lower, to=upper)
                print(f"Joint {i+1} range set to: [{lower:.2f}, {upper:.2f}] radians")
                
    def _on_joint_change(self, *args):
        """Joint slider change callback - core fix"""
        if not self.running:
            return
            
        # Get target joint positions
        target_positions = [var.get() for var in self.joint_vars]
        
        # Apply position control - directly set absolute positions
        self.apply_position_control(target_positions)
                
    def _on_target_change(self, *args):
        """Target position change callback"""
        self.target_position = np.array([var.get() for var in self.target_vars])
        self._create_target_visual()
        
    def _on_water_toggle(self):
        """Water current toggle callback"""
        self.current_variation = self.water_var.get()
        
    def _on_turbulence_change(self, *args):
        """Turbulence intensity change callback"""
        self.turbulence_strength = self.turbulence_var.get()
        
    def _reset_target(self):
        """Reset target position"""
        # Randomly generate new target position
        r = np.random.uniform(0.15, self.workspace_radius)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi) + self.base_height
        z = np.clip(z, 0.1, 0.7)
        
        self.target_position = np.array([x, y, z])
        
        # Update GUI
        for i, var in enumerate(self.target_vars):
            var.set(self.target_position[i])
            
        self._create_target_visual()
        self.success_achieved = False
        
    def _reset_to_home(self):
        """Reset robotic arm to home position"""
        home_positions = [0.0, 0.0, 0.0, 0.0]
        
        # Reset physics state
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.resetJointState(self.robot_id, joint_idx, home_positions[i])
        
        # Update GUI sliders
        for i, var in enumerate(self.joint_vars):
            var.set(home_positions[i])
        
        self.success_achieved = False
        
    def _start_simulation(self):
        """Start simulation"""
        if not self.running:
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
            # Start simulation thread
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
                # Apply underwater forces
                self._apply_underwater_forces()
                
                # Run physics simulation
                p.stepSimulation()
                
                # Check success status
                self._check_success()
                
                # Update status display
                self._update_status_display()
                
                # Increment time step
                self.time_step += 1
                
                # Control frame rate
                time.sleep(1/240.0)
                
            except Exception as e:
                print(f"Simulation error: {e}")
                break
                
    def _update_status_display(self):
        """Update status display"""
        def update():
            # Update distance
            self.distance_label.config(text=f"Distance to target: {self.current_distance:.3f}m")
            
            # Update success status
            if self.success_achieved:
                self.success_label.config(text="Status: Successfully reached!", foreground="green")
            else:
                self.success_label.config(text="Status: Not reached", foreground="red")
            
            # Update water current information
            if hasattr(self, 'current_velocity_actual'):
                water_speed = np.linalg.norm(self.current_velocity_actual)
                self.water_label.config(text=f"Water current speed: {water_speed:.3f}m/s")
            
            # Update joint position information
            joint_pos = self.get_main_joint_positions()
            joint_text = " ".join([f"{pos:.2f}" for pos in joint_pos])
            self.joint_pos_label.config(text=f"Joint positions: [{joint_text}]")
        
        # Update GUI in main thread
        self.root.after(0, update)
        
    def run(self):
        """Run application"""
        print("Starting underwater robotic arm manual control interface...")
        print("Use sliders to control robotic arm joints, try to reach the orange target sphere!")
        print("Observe underwater physics effects: fluid resistance, buoyancy, water current, etc.")
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("User interrupted")
        finally:
            self._cleanup()
            
    def _cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass

if __name__ == "__main__":
    print("=" * 60)
    print("Underwater Alpha Robotic Arm Manual Control Test")
    print("=" * 60)
    print("This is a test program for verifying underwater environment simulation")
    print("Features:")
    print("1. Manual control of robotic arm joints")
    print("2. Observe underwater physics effects")
    print("3. Verify target reaching task")
    print("4. Adjust water current parameters")
    print("=" * 60)
    
    try:
        app = UnderwaterManualControl()
        app.run()
    except Exception as e:
        print(f"Program error: {e}")
        import traceback
        traceback.print_exc()