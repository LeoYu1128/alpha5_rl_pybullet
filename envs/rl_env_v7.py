import os
# Set environment variable to avoid OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

class AlphaReachEnv(gym.Env):
    """
    Underwater Alpha Robotic Arm Reaching Task Environment
    Task: Control 4 main joints in underwater environment to make end effector reach target position
    Considers fluid drag, buoyancy, water current disturbances and other underwater physics characteristics
    """
    
    def __init__(self, render_mode=None, max_steps=500, reward_type='dense',
             enable_target_drift=True, 
             enable_domain_randomization=True, 
             enable_sensor_noise=True,
             enable_curriculum=True):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.physics_client = None
        self.reward_type = reward_type  # 'dense' or 'sparse'
        
        # ========== New: Domain randomization and curriculum learning control ==========
        self.enable_domain_randomization = enable_domain_randomization
        self.enable_curriculum = enable_curriculum
        self.enable_target_drift = enable_target_drift  # ← New: Target drift switch
        self.enable_sensor_noise = enable_sensor_noise
        self.curriculum_level = 0  # Curriculum level 0-5
        self.episode_count = 0  # Episode counter
        self.curriculum_stage = self.curriculum_level  # ✅ Add this line
        self.success_count = 0  # Success count (for curriculum advancement)
        # ================================================
        
        # Underwater environment parameters (baseline values, will vary in domain randomization)
        self.water_density = 1000.0  # Water density kg/m³
        self.gravity = -9.81  # Gravitational acceleration
        self.base_height = 0.1
        self.workspace_radius = 0.4
        
        # Fluid dynamics parameters
        self.drag_coefficient = 0.5  # Drag coefficient
        self.added_mass_coefficient = 0.3  # Added mass coefficient
        self.buoyancy_enabled = True  # Enable buoyancy
        
        # ========== New: Domain randomization parameter ranges ==========
        # Each reset, these parameters will be randomly sampled within baseline ± variation range
        self.dr_ranges = {
            'drag_coefficient': (0.3, 0.7),  # Drag coefficient range
            'water_density': (950, 1050),  # Water density range
            'current_velocity_scale': (0.5, 1.5),  # Water current velocity scale
            'turbulence_strength': (0.05, 0.2),  # Turbulence strength
            'mass_scale': (0.9, 1.1),  # Mass scale
            'joint_friction': (0.05, 0.15),  # Joint friction
        }
        # ============================================
        
        # ========== New: Real physics mass parameters ==========
        self.urdf_total_mass = 1.52              # Total mass in URDF (kg)
        self.target_actual_mass = 1.36           # Actual robotic arm mass (kg) from official data. Can be found in manual online
        self.target_underwater_mass = 0.9        # Underwater effective mass (kg)
        buoyancy_mass = self.target_actual_mass - self.target_underwater_mass
        self.buoyancy_compensation_ratio = buoyancy_mass / self.target_actual_mass
        self.robot_mass_scale = self.target_actual_mass / self.urdf_total_mass
        print(f"[Physics Parameters] Mass scale: {self.robot_mass_scale:.4f}, Buoyancy compensation: {self.buoyancy_compensation_ratio:.4f}")
        # ==========================================
        
        # Water current parameters
        self.current_velocity = np.array([0.1, 0.05, 0.0])  # Water current velocity m/s
        self.current_variation = True  # Whether water current varies
        self.turbulence_strength = 0.1  # Turbulence strength
        
        # ========== New: Curriculum learning configuration ==========
        self.curriculum_config = {
            0: {'workspace_radius': 0.4, 'success_threshold': 0.04, 'episodes_to_advance': 50, 'drift_strength': 0.0253},  # 1.0cm
            1: {'workspace_radius': 0.4, 'success_threshold': 0.04, 'episodes_to_advance': 50, 'drift_strength': 0.0355},  # 1.4cm
            2: {'workspace_radius': 0.4, 'success_threshold': 0.04, 'episodes_to_advance': 50, 'drift_strength': 0.0456},  # 1.8cm
            3: {'workspace_radius': 0.4, 'success_threshold': 0.04, 'episodes_to_advance': 50, 'drift_strength': 0.0558},  # 2.2cm
            4: {'workspace_radius': 0.4, 'success_threshold': 0.04, 'episodes_to_advance': 50, 'drift_strength': 0.0659},  # 2.6cm
            5: {'workspace_radius': 0.4, 'success_threshold': 0.04, 'episodes_to_advance': float('inf'), 'drift_strength': 0.0760},  # 3.0cm
        }
        # =========================================
        
        # ========== New: Target drift parameters ==========
        self.target_drift_damping = 0.90  # Damping coefficient, prevents drift from being too fast
        self.target_velocity = np.zeros(3, dtype=np.float32)  # Target current velocity
        self.max_drift_distance = 0.4  # Maximum drift distance (not exceeding workspace)
        self.initial_target_position = None  # Record initial target position
        # =======================================
        # ========== New: Sensor noise parameters ==========
        self.enable_sensor_noise = True              # Enable noise
        self.position_noise_std = 0.002              # Joint position noise std (about 0.2%)
        self.velocity_noise_std = 0.008              # Joint velocity noise std (about 0.8%)
        self.ee_position_noise_std = 0.003           # End effector position noise std (0.3cm)
        print(f"[Sensor Noise] Position: {self.position_noise_std}, Velocity: {self.velocity_noise_std}, End-effector: {self.ee_position_noise_std}")
        # ==========================================

        # Connect physics engine
        self._connect_physics()
        self._setup_underwater_scene()
        self._analyze_robot()

        # Reward shaping coefficients
        self.distance_scale = 1.0
        self.progress_scale = 1.4
        self.control_penalty = 0.01
        self.velocity_penalty = 0.02
        self.time_penalty = 0.005
        self.success_bonus = 10.0
        self.milestone_bonus = 5.0
        self.milestone_thresholds = 0.05
        self.success_threshold = 0.03
        self.previous_distance = None

        # Define action space - Position increments for 4 main joints (normalized to [-1,1])
        self.action_space = spaces.Box(
            low=-1.0,   # Normalized action space
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Define observation space - [4 joint positions, 4 joint velocities, 3 end effector position, 3 target position]
        obs_dim = 4 + 4 + 3 + 3  # 14 dimensions 
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        print("Underwater Alpha robotic arm environment initialization complete")
        print(f"Action space: {self.action_space}")
        print(f"Observation space: {self.observation_space.shape}")
        print(f"Water current velocity: {self.current_velocity}")
        print(f"Domain randomization: {'Enabled' if self.enable_domain_randomization else 'Disabled'}")
        print(f"Curriculum learning: {'Enabled' if self.enable_curriculum else 'Disabled'} (Current level: {self.curriculum_level})")
        print(f"Target drift: {'Enabled' if self.enable_target_drift else 'Disabled'}")
    
    def _connect_physics(self):
        """Connect PyBullet physics engine"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
        
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(
                p.COV_ENABLE_MOUSE_PICKING, 0, 
                physicsClientId=self.physics_client
            )
            # Set underwater viewpoint
            p.resetDebugVisualizerCamera(
                2.0, 30, -20, [0, 0, 0.3], 
                physicsClientId=self.physics_client
            )
            # Set blue background to simulate underwater environment
            p.configureDebugVisualizer(
                p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1,
                physicsClientId=self.physics_client
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Set gray background (instead of white)
        p.configureDebugVisualizer(
            p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1,
            physicsClientId=self.physics_client
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, 1,
            physicsClientId=self.physics_client
        )

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        
        # Set gravity (gravity effect is reduced in underwater environment)
        p.setGravity(0, 0, self.gravity * 0.1, physicsClientId=self.physics_client)  # Reduce gravity effect
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
    
    def _setup_underwater_scene(self):
        """Setup underwater simulation scene"""
        # Load seabed ground
        self.plane_id = p.loadURDF(
            "plane.urdf", 
            physicsClientId=self.physics_client
        )
        
        # Set ground to dark blue to simulate seabed
        p.changeVisualShape(
            self.plane_id, -1, 
            rgbaColor=[0.1, 0.2, 0.4, 1.0],
            physicsClientId=self.physics_client
        )
        
        # Load Alpha robotic arm
        # Compatible with different working directories and GUI/DIRECT mode URDF search
        try:
            from pathlib import Path
            this_dir = Path(__file__).resolve().parent
            proj_root = this_dir.parent  # Project root directory (contains alpha_description/)
        except Exception:
            this_dir = None
            proj_root = None

        robot_paths = [
            # Relative path candidates
            "alpha_robot_for_pybullet.urdf",
            "alpha_description/urdf/alpha_robot_for_pybullet.urdf",
            "../alpha_description/urdf/alpha_robot_for_pybullet.urdf",
        ]
        # Absolute path candidates (based on current file and project root)
        if proj_root is not None:
            robot_paths.extend([
                str((proj_root / "alpha_description/urdf/alpha_robot_for_pybullet.urdf").resolve()),
                str((proj_root.parent / "alpha_description/urdf/alpha_robot_for_pybullet.urdf").resolve()),
            ])
        
        self.robot_id = None
        for robot_path in robot_paths:
            if os.path.exists(robot_path):
                try:
                    self.robot_id = p.loadURDF(
                        robot_path,
                        basePosition=[0, 0, self.base_height],
                        useFixedBase=True,
                        physicsClientId=self.physics_client
                    )
                    print(f"Successfully loaded underwater Alpha robotic arm: {robot_path}")
                    break
                except Exception as e:
                    print(f"Failed to load {robot_path}: {e}")
                    continue
        
        if self.robot_id is None:
            raise FileNotFoundError("Cannot find Alpha robotic arm URDF file")
        
        # Add some underwater decorations
        self._add_underwater_decorations()

    def _add_underwater_decorations(self):
        """Add underwater decorations"""
        try:
            # Add some spheres as underwater obstacles/decorations
            for i in range(3):
                x = np.random.uniform(-0.8, 0.8)
                y = np.random.uniform(-0.8, 0.8)
                z = np.random.uniform(0.1, 0.5)
                
                decoration_id = p.loadURDF(
                    "sphere_small.urdf",
                    basePosition=[x, y, z],
                    physicsClientId=self.physics_client
                )
                
                # Set to dark green to simulate seaweed or reef
                p.changeVisualShape(
                    decoration_id, -1, 
                    rgbaColor=[0.1, 0.4, 0.2, 0.8],
                    physicsClientId=self.physics_client
                )
        except:
            print("Cannot load decorations, skipping...")
    
    def _analyze_robot(self):
        """Analyze robot structure, extract joint information and set underwater physics properties"""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        # Store joint information
        self.joint_info = {}
        self.main_joint_indices = []
        
        # Iterate through all joints
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            self.joint_info[i] = {
                'name': joint_name,
                'type': joint_type,
                'lower': lower_limit,
                'upper': upper_limit
            }
            
            # Identify main control joints
            if joint_name in ['joint_1', 'joint_2', 'joint_3', 'joint_4']:
                self.main_joint_indices.append(i)
        
        print(f"Found main control joints: {len(self.main_joint_indices)}")
        for i, joint_idx in enumerate(self.main_joint_indices):
            joint = self.joint_info[joint_idx]
            print(f"  [{i}] {joint['name']}: [{joint['lower']:.2f}, {joint['upper']:.2f}] radians")
        
        # Find end effector
        self.tcp_index = None
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            if 'tcp' in joint_name.lower():
                self.tcp_index = i
                break
        
        if self.tcp_index is None:
            self.tcp_index = num_joints - 1
        
        print(f"End effector index: {self.tcp_index}")
        
        # Set underwater physics properties
        self._setup_underwater_dynamics()
    
    def _setup_underwater_dynamics(self):
        """
        Setup underwater dynamics properties
        - Scale mass to real values
        - Set underwater damping
        - Record link masses for subsequent buoyancy calculations
        """
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        # Record real masses of each link (for buoyancy calculation)
        self.link_masses = {}
        self.link_indices = []
        
        total_urdf_mass = 0.0
        total_scaled_mass = 0.0
        
        # Iterate through all links (-1 is base link)
        for i in range(-1, num_joints):
            try:
                # Get dynamics parameters defined in URDF
                dynamics_info = p.getDynamicsInfo(
                    self.robot_id, i, 
                    physicsClientId=self.physics_client
                )
                urdf_mass = dynamics_info[0]
                total_urdf_mass += urdf_mass
                
                # If link has mass, scale to real value
                if urdf_mass > 0:
                    # Scale mass
                    real_mass = urdf_mass * self.robot_mass_scale
                    self.link_masses[i] = real_mass
                    self.link_indices.append(i)
                    total_scaled_mass += real_mass
                    
                    # Apply real mass to simulation
                    p.changeDynamics(
                        self.robot_id, 
                        i, 
                        mass=real_mass,
                        physicsClientId=self.physics_client
                    )
                
                # Set underwater damping parameters
                if i >= 0:  # Joint links
                    p.changeDynamics(
                        self.robot_id, 
                        i, 
                        linearDamping=2.0,      # Linear damping
                        angularDamping=2.0,     # Angular damping
                        jointDamping=0.5,       # Joint damping
                        physicsClientId=self.physics_client
                    )
                else:  # base link
                    p.changeDynamics(
                        self.robot_id, 
                        i, 
                        linearDamping=1.0, 
                        angularDamping=1.0,
                        physicsClientId=self.physics_client
                    )
            
            except Exception as e:
                # Some links may not have dynamics info, skip
                pass
        
        # Verify mass scaling is correct
        print(f"[Mass Scaling] URDF total mass: {total_urdf_mass:.4f} kg")
        print(f"[Mass Scaling] Scaled total mass: {total_scaled_mass:.4f} kg (Target: {self.target_actual_mass:.4f} kg)")
        print(f"[Mass Scaling] Number of links with mass: {len(self.link_indices)}")
    
    def _update_current_velocity(self):
        """Update water current velocity (simulate varying water current)"""
        if self.current_variation:
            # Add time-varying water current and turbulence
            time_factor = self.current_step * 0.01
            
            # Base current + periodic variation + random turbulence
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
        """
        Apply underwater mechanical effects
        1. Buoyancy - Counteracts 33.8% of gravity
        2. Fluid drag - Resists end effector motion
        """
        self._update_current_velocity()
        
        # ========== 1. Apply buoyancy to all links ==========
        if self.buoyancy_enabled:
            for link_idx in self.link_indices:
                try:
                    mass = self.link_masses.get(link_idx, 0)
                    if mass > 0:
                        # Calculate gravitational force on this link (downward)
                        gravity_force = mass * abs(self.gravity)  # N
                        
                        # Calculate buoyancy (upward, counteracts 33.8% of gravity)
                        buoyancy_force = gravity_force * self.buoyancy_compensation_ratio
                        
                        # Apply buoyancy (Z-axis upward)
                        p.applyExternalForce(
                            self.robot_id, 
                            link_idx,
                            forceObj=[0, 0, buoyancy_force],
                            posObj=[0, 0, 0],
                            flags=p.LINK_FRAME,
                            physicsClientId=self.physics_client
                        )
                except:
                    pass
        
        # ========== 2. Apply fluid drag to end effector ==========
        try:
            ee_state = p.getLinkState(
                self.robot_id, 
                self.tcp_index, 
                computeLinkVelocity=1,
                physicsClientId=self.physics_client
            )
            ee_velocity = np.array(ee_state[6])  # Linear velocity
            
            # Calculate velocity relative to water current
            relative_velocity = ee_velocity - self.current_velocity_actual
            
            # Calculate fluid drag: F = -0.5 * ρ * Cd * A * v * |v|
            drag_force = -0.5 * self.water_density * self.drag_coefficient * 0.01 * \
                         relative_velocity * np.linalg.norm(relative_velocity)
            
            # Limit force magnitude to avoid numerical instability
            max_force = 5.0
            drag_force = np.clip(drag_force, -max_force, max_force)
            
            # Apply drag
            p.applyExternalForce(
                self.robot_id, 
                self.tcp_index,
                forceObj=drag_force,
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.physics_client
            )
        except:
            pass
    
    
    def _sample_target_position(self):
        """Sample target position within robotic arm reachable range (supports curriculum learning)"""
        # Get current curriculum level parameters
        curriculum_params = self._get_curriculum_params()
        workspace_radius = curriculum_params['workspace_radius']
        
        max_attempts = 50

        # Adjust reachable range based on curriculum level
        safe_radius_min = 0.12       # Minimum distance
        safe_radius_max = min(0.25, workspace_radius)  # Use curriculum level workspace_radius
        z_min = 0.15                 # Height lower bound
        z_max = 0.35                 # Height upper bound

        for _ in range(max_attempts):
            # Sample within cylinder (better fits robotic arm reachable range)
            r = np.random.uniform(safe_radius_min, safe_radius_max)
            theta = np.random.uniform(-np.pi/2, np.pi/2)  # Limited to frontal fan-shaped region
            z = np.random.uniform(z_min, z_max)

            x = r * np.cos(theta)
            y = r * np.sin(theta)

            return np.array([x, y, z], dtype=np.float32)

        # If sampling fails, return a guaranteed reachable position
        return np.array([0.2, 0.0, 0.25], dtype=np.float32)
    
    def _apply_domain_randomization(self):
        """Apply domain randomization - Randomize physics parameters"""
        if not self.enable_domain_randomization:
            return
        
        # Randomize drag coefficient
        self.drag_coefficient = np.random.uniform(*self.dr_ranges['drag_coefficient'])
        
        # Randomize water density
        self.water_density = np.random.uniform(*self.dr_ranges['water_density'])
        
        # Randomize water current velocity
        current_scale = np.random.uniform(*self.dr_ranges['current_velocity_scale'])
        self.current_velocity_actual = self.current_velocity * current_scale
        
        # Randomize turbulence strength
        self.turbulence_strength = np.random.uniform(*self.dr_ranges['turbulence_strength'])
        
        # Randomize robot mass
        mass_scale = np.random.uniform(*self.dr_ranges['mass_scale'])
        for link_idx in self.link_indices:
            try:
                dynamics = p.getDynamicsInfo(self.robot_id, link_idx, physicsClientId=self.physics_client)
                base_mass = dynamics[0]
                if base_mass > 0:
                    new_mass = base_mass * mass_scale
                    p.changeDynamics(
                        self.robot_id, link_idx,
                        mass=new_mass,
                        physicsClientId=self.physics_client
                    )
            except:
                pass
        
        # Randomize joint friction
        joint_friction = np.random.uniform(*self.dr_ranges['joint_friction'])
        for joint_idx in self.main_joint_indices:
            p.changeDynamics(
                self.robot_id, joint_idx,
                jointDamping=joint_friction,
                physicsClientId=self.physics_client
            )
        
        if self.episode_count % 100 == 0:  # Print every 100 episodes
            print(f"[Domain Randomization] Drag={self.drag_coefficient:.3f}, Water density={self.water_density:.0f}, "
                  f"Current scale={current_scale:.2f}, Turbulence={self.turbulence_strength:.3f}")
    
    def _update_curriculum(self, success):
        """Update curriculum learning level"""
        if not self.enable_curriculum:
            return
        
        # Count successes
        if success:
            self.success_count += 1
        
        # Check if advancement is needed
        current_config = self.curriculum_config[self.curriculum_level]
        episodes_needed = current_config['episodes_to_advance']
        
        # Calculate success rate (last episodes_needed episodes)
        if self.episode_count % episodes_needed == 0 and self.episode_count > 0:
            success_rate = self.success_count / episodes_needed
            
            # Success rate exceeds 70% and not at highest level, advance
            if success_rate > 0.7 and self.curriculum_level < 5:
                self.curriculum_level += 1
                self.curriculum_stage = self.curriculum_level  # ← New
                self.success_count = 0  # Reset success count
                new_config = self.curriculum_config[self.curriculum_level]
                
                # Update environment parameters
                self.workspace_radius = new_config['workspace_radius']
                self.success_threshold = new_config['success_threshold']
                
                print(f"\n{'='*60}")
                print(f"[Curriculum Advance] Level {self.curriculum_level-1} -> {self.curriculum_level}")
                print(f"  Success rate: {success_rate*100:.1f}% (last {episodes_needed} episodes)")
                print(f"  New workspace radius: {self.workspace_radius:.2f}m")
                print(f"  New success threshold: {self.success_threshold:.3f}m")
                print(f"{'='*60}\n")
            else:
                # Success rate insufficient, reset count and continue current level
                self.success_count = 0
                if self.curriculum_level < 5:
                    print(f"[Curriculum Learning] Level {self.curriculum_level} success rate {success_rate*100:.1f}% - Continue training")
    
    def _get_curriculum_params(self):
        """Get current curriculum level parameters"""
        if not self.enable_curriculum:
            return {
                'workspace_radius': self.workspace_radius,
                'success_threshold': self.success_threshold,
                'drift_strength': 0.05  # Default maximum drift strength
            }
        
        config = self.curriculum_config[self.curriculum_level]
        return {
            'workspace_radius': config['workspace_radius'],
            'success_threshold': config['success_threshold'],
            'drift_strength': config['drift_strength']  # Return drift strength
        }
    
    def _update_target_drift(self):
        """Update target position drift (affected by water current)"""
        if not self.enable_target_drift:
            return
        
        # Get current curriculum level drift strength
        curriculum_params = self._get_curriculum_params()
        drift_strength = curriculum_params['drift_strength']
        
        # Calculate water current effect on target (with drift strength scaling)
        water_force = self.current_velocity_actual * drift_strength * 0.1  # 0.1 is base scaling coefficient
        
        # Update target velocity (simplified Newton's second law)
        self.target_velocity += water_force
        
        # Apply damping (prevent velocity from growing infinitely)
        self.target_velocity *= self.target_drift_damping
        
        # Update target position
        new_target_position = self.target_position + self.target_velocity * 0.01  # Time step
        
        # ========== Key: Constrain within workspace ==========
        # Calculate distance from initial position
        distance_from_initial = np.linalg.norm(new_target_position - self.initial_target_position)
        
        if distance_from_initial > self.max_drift_distance:
            # If out of range, pull target back to boundary
            direction = (new_target_position - self.initial_target_position) / distance_from_initial
            new_target_position = self.initial_target_position + direction * self.max_drift_distance
            # Bounce velocity (simplified collision handling)
            self.target_velocity *= -0.5
        
        # Update target position
        self.target_position = new_target_position
        
        # Update target visualization position
        if hasattr(self, 'target_visual_id'):
            try:
                p.resetBasePositionAndOrientation(
                    self.target_visual_id,
                    self.target_position,
                    [0, 0, 0, 1],
                    physicsClientId=self.physics_client
                )
            except:
                pass

    def _get_joint_positions(self):
        """Get current positions of main joints (with sensor noise)"""
        positions = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(
                self.robot_id, joint_idx,
                physicsClientId=self.physics_client
            )
            true_position = joint_state[0]  # True position
            
            # Add Gaussian noise
            if self.enable_sensor_noise:
                noise = np.random.normal(0, self.position_noise_std)
                noisy_position = true_position + noise
                
                # Ensure within joint limits
                joint_info = self.joint_info[joint_idx]
                noisy_position = np.clip(
                    noisy_position, 
                    joint_info['lower'], 
                    joint_info['upper']
                )
            else:
                noisy_position = true_position
            
            positions.append(noisy_position)
        
        return np.array(positions, dtype=np.float32)
    
    def _get_joint_velocities(self):
        """Get current velocities of main joints (with sensor noise)"""
        velocities = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(
                self.robot_id, joint_idx,
                physicsClientId=self.physics_client
            )
            true_velocity = joint_state[1]  # True velocity
            
            # Add Gaussian noise
            if self.enable_sensor_noise:
                noise = np.random.normal(0, self.velocity_noise_std)
                noisy_velocity = true_velocity + noise
            else:
                noisy_velocity = true_velocity
            
            velocities.append(noisy_velocity)
        
        return np.array(velocities, dtype=np.float32)
    
    def _get_end_effector_position(self):
        """Get end effector position (with sensor noise)"""
        link_state = p.getLinkState(
            self.robot_id, self.tcp_index,
            physicsClientId=self.physics_client
        )
        true_position = np.array(link_state[0], dtype=np.float32)
        
        # Add 3D Gaussian noise
        if self.enable_sensor_noise:
            noise = np.random.normal(0, self.ee_position_noise_std, size=3)
            noisy_position = true_position + noise
        else:
            noisy_position = true_position
        
        return noisy_position
    
    def _apply_action(self, action):
        """Apply action to underwater robotic arm"""
        # Get current joint positions
        current_positions = self._get_joint_positions()
        ee_pos = self._get_end_effector_position()
        current_distance = np.linalg.norm(ee_pos - self.target_position)
        # Scale normalized action to actual increment (increase action magnitude)
        if current_distance > 0.15:
            scale = 0.15
        elif current_distance > 0.08:  
            scale = 0.06
        elif current_distance > 0.05:  # current_distance between 0.04 and 0.08
            scale = 0.03
        elif current_distance > 0.03:
            scale = 0.02
        else:
            scale = 0.01

        scaled_action = action * scale
        target_positions = current_positions + scaled_action
        
        # Apply joint limits
        for i, joint_idx in enumerate(self.main_joint_indices):
            joint = self.joint_info[joint_idx]
            target_positions[i] = np.clip(
                target_positions[i],
                joint['lower'],
                joint['upper']
            )
        
        # Execute position control (underwater environment has smaller forces and velocities)
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                maxVelocity=1.0,  # Slower speed underwater
                force=9,        # Larger force needed underwater to overcome drag
                physicsClientId=self.physics_client
            )
    

    def _compute_reward(self, action):
        """Reward combining distance, progress and cost (supports curriculum learning)"""
        achieved_goal = self._get_end_effector_position()
        desired_goal = self.target_position

        distance = float(np.linalg.norm(achieved_goal - desired_goal))
        prev_distance = self.previous_distance if self.previous_distance is not None else distance
        distance_delta = prev_distance - distance
        
        # Get current curriculum level success threshold
        curriculum_params = self._get_curriculum_params()
        success_threshold = curriculum_params['success_threshold']

        shaped_distance = -self.distance_scale * distance
        progress_reward = self.progress_scale * distance_delta
        control_cost = self.control_penalty * float(np.linalg.norm(action) ** 2)
        velocity_cost = self.velocity_penalty * float(np.linalg.norm(self._get_joint_velocities()))
        time_cost = self.time_penalty
        milestone_reward = self.milestone_bonus if distance < self.milestone_thresholds else 0.0
        success = distance < success_threshold  # Use curriculum level threshold

        total_reward = shaped_distance + progress_reward - control_cost - velocity_cost - time_cost + milestone_reward
        success_bonus = self.success_bonus if success else 0.0
        total_reward += success_bonus

        self.previous_distance = distance

        reward_terms = {
            'reward_distance': shaped_distance,
            'reward_progress': progress_reward,
            'reward_control': -control_cost,
            'reward_velocity': -velocity_cost,
            'reward_time': -time_cost,
            'reward_success': success_bonus
        }

        return float(total_reward), bool(success), reward_terms
    
    def _get_observation(self):
        """Get underwater environment observation (includes water current information)"""
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        ee_position = self._get_end_effector_position()
        
        # Combine observation vector (add water current velocity information)
        observation = np.concatenate([
            joint_positions,              # 4 dimensions
            joint_velocities,             # 4 dimensions  
            ee_position,                  # 3 dimensions
            self.target_position,         # 3 dimensions
        ]).astype(np.float32)
        
        return observation
    
    def _create_target_visual(self):
        """Create target position visual marker (underwater style)"""
        if hasattr(self, 'target_visual_id'):
            try:
                p.removeBody(self.target_visual_id, physicsClientId=self.physics_client)
            except:
                pass
        
        # Create orange sphere as target marker (more visible underwater)
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02,  # Restored to half of original size
            rgbaColor=[1, 0.5, 0, 1.0],  # Orange, fully opaque
            physicsClientId=self.physics_client
        )
        
        self.target_visual_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_position,
            physicsClientId=self.physics_client
        )
    
    def reset(self, seed=None, options=None):
        """Reset underwater environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Ensure physics connection exists
        if self.physics_client is None:
            self._connect_physics()
            self._setup_underwater_scene()
            self._analyze_robot()
        
        self.current_step = 0
        self.episode_count += 1  # Increment episode count
        
        # ========== New: Apply domain randomization ==========
        self._apply_domain_randomization()
        # ======================================
        
        # Random initial joint positions
        init_positions = []
        for joint_idx in self.main_joint_indices:
            joint = self.joint_info[joint_idx]
            range_center = (joint['lower'] + joint['upper']) / 2
            range_width = (joint['upper'] - joint['lower']) * 0.3  # Smaller initialization range underwater
            init_pos = np.random.uniform(
                range_center - range_width,
                range_center + range_width
            )
            init_positions.append(init_pos)
        
        # Set joint initial positions
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.resetJointState(
                self.robot_id, joint_idx, init_positions[i],
                physicsClientId=self.physics_client
            )
        
        # Generate new target position
        self.target_position = self._sample_target_position()
        
        # ========== New: Initialize target drift parameters ==========
        self.initial_target_position = self.target_position.copy()  # Record initial position
        self.target_velocity = np.zeros(3, dtype=np.float32)  # Reset target velocity
        # ==========================================

        # Reset water current
        self.current_velocity_actual = self.current_velocity.copy()

        # Create target visualization (always create to be visible in GIF)
        self._create_target_visual()
        
        # Stabilize simulation
        for _ in range(100):  # Underwater needs longer time to stabilize
            p.stepSimulation(physicsClientId=self.physics_client)

        self.previous_distance = float(np.linalg.norm(
            self._get_end_effector_position() - self.target_position
        ))

        observation = self._get_observation()
        info = {
            'is_success': False,
            'underwater': True
        }
        return observation, info
    
    def step(self, action):
        """Execute one step of underwater action"""
        self.current_step += 1
        
        # Ensure action type is correct
        action = np.array(action, dtype=np.float32)
        
        # Apply action
        self._apply_action(action)
        
        # Apply underwater mechanical effects
        self._apply_underwater_forces()
        
        # ========== New: Update target drift ==========
        self._update_target_drift()
        # ====================================
        
        # Run physics simulation (reduce steps to improve responsiveness)
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward, success, reward_terms = self._compute_reward(action)
        
        # Check termination conditions
        terminated = bool(success)
        truncated = bool(self.current_step >= self.max_steps)
        
        # ========== New: Update curriculum at episode end ==========
        if terminated or truncated:
            self._update_curriculum(success)
        # =============================================
        
        # Calculate current distance
        ee_pos = self._get_end_effector_position()
        current_distance = float(np.linalg.norm(ee_pos - self.target_position))
        
        info = {
            'success': success,
            'distance': current_distance,
            'is_success': success,
            'current_velocity': self.current_velocity_actual.copy(),
            'underwater': True
        }

        info.update(reward_terms)

        return observation, reward, terminated, truncated, info  # Return terminated and truncated separately

    def close(self):
        """Close environment"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.physics_client = None

    
    def render(self):
        """Render underwater environment"""
        pass

# Test underwater environment
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Test 1: Basic environment (no domain randomization, no curriculum learning)")
    print("="*60)
    env = AlphaReachEnv(render_mode="human")
    obs, info = env.reset()
    print(f"Initial observation dimension: {obs[0].shape}")
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 10 == 0:
            print(f"Step {step}: Distance={info['distance']:.3f}m")
        if terminated or truncated:
            print(f"Episode ended: Success={info['success']}")
            break
    env.close()
    
    print("\n" + "="*60)
    print("Test 2: Enable domain randomization")
    print("="*60)
    env = AlphaReachEnv(render_mode="human", enable_domain_randomization=True)
    
    for episode in range(3):
        obs, info = env.reset()
        print(f"\nEpisode {episode+1}:")
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"  Ended: Success={info['success']}, Distance={info['distance']:.3f}m")
                break
    env.close()
    
    print("\n" + "="*60)
    print("Test 3: Enable curriculum learning (need multiple episodes to observe advancement)")
    print("="*60)
    env = AlphaReachEnv(render_mode="human", enable_curriculum=True)
    
    for episode in range(10):
        obs, info = env.reset()
        print(f"\nEpisode {episode+1} (Curriculum level: {env.curriculum_level}):")
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"  Ended: Success={info['success']}, Distance={info['distance']:.3f}m")
                break
    env.close()
    
    print("\n" + "="*60)
    print("Test 4: Enable both domain randomization and curriculum learning")
    print("="*60)
    env = AlphaReachEnv(
        render_mode="human", 
        enable_domain_randomization=True,
        enable_curriculum=True
    )
    
    for episode in range(5):
        obs, info = env.reset()
        print(f"\nEpisode {episode+1} (Curriculum level: {env.curriculum_level}):")
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"  Ended: Success={info['success']}, Distance={info['distance']:.3f}m")
                break
    env.close()
    
    print("\nUnderwater environment test complete!")