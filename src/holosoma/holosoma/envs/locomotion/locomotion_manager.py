from __future__ import annotations

from loguru import logger

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_utils import torch_rand_float


class LeggedRobotLocomotionManager(BaseTask):
    BASE_NUM_ENVS = 4096

    def __init__(
        self,
        tyro_config,
        *,
        device,
    ):
        self.init_done = False
        super().__init__(
            tyro_config,
            device=device,
        )

        # Setup link groups and masses for momentum computation
        self._setup_momentum_computation()

        self.init_done = True

    def _setup_momentum_computation(self):
        """Setup link groups and cache masses for angular momentum computation."""

        # ================== Body Structure =================
        # body_names: ['Trunk', 'H1', 'H2', 'AL1', 'AL2', 'AL3', 'AL4', 'AL5', 'AL6', 'left_hand_link', 'AR1', 'AR2', 'AR3', 'AR4', 'AR5', 'AR6', 'right_hand_link', 'Waist', 'Hip_Pitch_Left', 'Hip_Roll_Left', 'Hip_Yaw_Left', 'Shank_Left', 'Ankle_Cross_Left', 'left_foot_link', 'left_foot_contact_point', 'Hip_Pitch_Right', 'Hip_Roll_Right', 'Hip_Yaw_Right', 'Shank_Right', 'Ankle_Cross_Right', 'right_foot_link', 'right_foot_contact_point']
        # Number of bodies: 32
        # ================== Mass Properties =================
        # Body 0 (Trunk): mass=12.224
        # Body 1 (H1): mass=0.444
        # Body 2 (H2): mass=0.631
        # Body 3 (AL1): mass=0.530
        # Body 4 (AL2): mass=0.160
        # Body 5 (AL3): mass=1.020
        # Body 6 (AL4): mass=0.298
        # Body 7 (AL5): mass=0.422
        # Body 8 (AL6): mass=0.265
        # Body 9 (left_hand_link): mass=0.249
        # Body 10 (AR1): mass=0.530
        # Body 11 (AR2): mass=0.160
        # Body 12 (AR3): mass=1.020
        # Body 13 (AR4): mass=0.360
        # Body 14 (AR5): mass=0.434
        # Body 15 (AR6): mass=0.282
        # Body 16 (right_hand_link): mass=0.246
        # Body 17 (Waist): mass=2.581
        # Body 18 (Hip_Pitch_Left): mass=0.944
        # Body 19 (Hip_Roll_Left): mass=0.395
        # Body 20 (Hip_Yaw_Left): mass=2.431
        # Body 21 (Shank_Left): mass=1.566
        # Body 22 (Ankle_Cross_Left): mass=0.072
        # Body 23 (left_foot_link): mass=0.818
        # Body 24 (left_foot_contact_point): mass=0.001
        # Body 25 (Hip_Pitch_Right): mass=0.939
        # Body 26 (Hip_Roll_Right): mass=0.349
        # Body 27 (Hip_Yaw_Right): mass=2.056
        # Body 28 (Shank_Right): mass=2.043
        # Body 29 (Ankle_Cross_Right): mass=0.078
        # Body 30 (right_foot_link): mass=0.765
        # Body 31 (right_foot_contact_point): mass=0.001
        
        # Define link indices for momentum computation
        # Based on T1 robot structure: AL* = left arm, AR* = right arm
        self.torso_indices = [0, 1, 2, 17]  # Trunk, H1, H2, Waist
        self.left_arm_indices = [3, 4, 5, 6, 7, 8, 9]  # AL1-6, left_hand_link
        self.right_arm_indices = [10, 11, 12, 13, 14, 15, 16]  # AR1-6, right_hand_link
        self.left_leg_indices = [18, 19, 20, 21, 22, 23]  # Left leg Hip_Pitch/Roll/Yaw, Shank, Ankle, foot (exclude contact point)
        self.right_leg_indices = [25, 26, 27, 28, 29, 30]  # Right leg
        
        # Cache masses for all bodies
        rb_props = self.simulator.gym.get_actor_rigid_body_properties(
            self.simulator.envs[0],
            self.simulator.robot_handles[0]
        )
        
        # Store as tensors on device for efficient computation
        masses = torch.tensor([prop.mass for prop in rb_props], 
                            dtype=torch.float32, device=self.device)
        
        # Create mass tensors for each group (shape: [num_links_in_group])
        self.torso_masses = masses[self.torso_indices]
        self.left_arm_masses = masses[self.left_arm_indices]
        self.right_arm_masses = masses[self.right_arm_indices]
        self.left_leg_masses = masses[self.left_leg_indices]
        self.right_leg_masses = masses[self.right_leg_indices]
        
        # Total mass for each group (for potential normalization)
        self.total_torso_mass = self.torso_masses.sum().item()
        self.total_left_arm_mass = self.left_arm_masses.sum().item()
        self.total_right_arm_mass = self.right_arm_masses.sum().item()
        self.total_left_leg_mass = self.left_leg_masses.sum().item()
        self.total_right_leg_mass = self.right_leg_masses.sum().item()

    def _get_task_name(self) -> str:
        return "locomotion"

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()

        self.base_quat = self.simulator.base_quat

        # initialize some data used later on
        self._init_counters()
        # joint positions offsets
        self.default_dof_pos_base = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if name not in self.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")
            angle = self.robot_config.init_state.default_joint_angles[name]
            self.default_dof_pos_base[i] = angle

        self.default_dof_pos_base = self.default_dof_pos_base.unsqueeze(0)  # (1, num_dof)
        self.default_dof_pos = self.default_dof_pos_base.repeat(self.num_envs, 1).clone()  # (num_envs, num_dof)

        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self._init_domain_rand_buffers()

        self.lidar_height_offset = getattr(self.robot_config, "lidar_height_offset", 0.5)

    def _init_counters(self):
        self.common_step_counter = 0

    def _update_counters_each_step(self):
        self.common_step_counter += 1

    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        # Action delay buffers are now initialized by randomization manager's setup_action_delay_buffers term

        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.record_push_robot_vel_buf = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._randomize_push_robots = False
        self._max_push_vel = torch.zeros(2, dtype=torch.float32, device=self.device)

    def _setup_robot_body_indices(self):
        foot_body_names = [s for s in self.body_names if self.robot_config.foot_body_name in s]
        foot_height_names = [s for s in self.body_names if self.robot_config.foot_height_name in s]

        termination_contact_names = []
        for name in self.robot_config.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        self.feet_indices = torch.zeros(len(foot_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(foot_body_names):
            self.feet_indices[i] = self.simulator.find_rigid_body_indice(name)

        self.feet_height_indices = torch.zeros(
            len(foot_height_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i, name in enumerate(foot_height_names):
            self.feet_height_indices[i] = self.simulator.find_rigid_body_indice(name)

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.simulator.find_rigid_body_indice(termination_contact_names[i])

        if self.robot_config.has_torso:
            self.torso_name = self.robot_config.torso_name
            self.torso_index = self.simulator.find_rigid_body_indice(self.torso_name)

    def set_is_evaluating(self, command=None):
        logger.info("Setting Env is evaluating")
        super().set_is_evaluating()
        commands = self.command_manager.commands
        commands.zero_()
        if command is not None:
            command_tensor = torch.as_tensor(command, device=self.device, dtype=commands.dtype)
            commands[:] = command_tensor.view(1, -1).expand_as(commands)
        gait_state = self.command_manager.get_state("locomotion_gait")
        gait_state.set_eval_mode(True)

    def _setup_simulator_next_task(self):
        pass

    def _setup_simulator_control(self):
        self.simulator.commands = self.command_manager.commands

    def _get_envs_to_refresh(self):
        return self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()

    def _refresh_envs_after_reset(self, env_ids):
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
        self.simulator.clear_contact_forces_history(env_ids)
        self.need_to_refresh_envs[env_ids] = False
        self.simulator.refresh_sim_tensors()
        self._pre_compute_observations_callback()

    def _pre_compute_observations_callback(self):
        # prepare quantities
        self.base_quat[:] = self.simulator.base_quat[:]
        self.terrain_manager.update_heights()

    def _update_tasks_callback(self):
        super()._update_tasks_callback()

        # Assign commands to simulator for headless recording
        if hasattr(self.simulator, "headless_recording") and self.simulator.headless_recording:
            if hasattr(self.command_manager, "commands"):
                self.simulator.commands = self.command_manager.commands

    def _post_compute_observations_callback(self):
        return

    def reset_all(self):
        self._init_buffers()
        return super().reset_all()

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # if target_states is not None, reset to target states
        if target_states is not None:
            self._reset_dofs(env_ids, target_states["dof_states"])
            self._reset_root_states(env_ids, target_states["root_states"])
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        # Observation manager reset is now handled in base_task.py
        self.need_to_refresh_envs[env_ids] = True

        if target_buf is not None:
            self.simulator.dof_pos[env_ids] = target_buf["dof_pos"].to(self.simulator.dof_pos.dtype)
            self.simulator.dof_vel[env_ids] = target_buf["dof_vel"].to(self.simulator.dof_vel.dtype)
            self.base_quat[env_ids] = target_buf["base_quat"].to(self.base_quat.dtype)
            self.episode_length_buf[env_ids] = target_buf["episode_length_buf"].to(self.episode_length_buf.dtype)
            self.reset_buf[env_ids] = target_buf["reset_buf"].to(self.reset_buf.dtype)
            self.time_out_buf[env_ids] = target_buf["time_out_buf"].to(self.time_out_buf.dtype)
            self._pending_episode_update_mask[env_ids] = False
            self._pending_episode_lengths[env_ids] = 0
        else:
            self.episode_length_buf[env_ids] = 0
            self.reset_buf[env_ids] = 1
            self._pending_episode_update_mask[env_ids] = True

    def _update_log_dict(self):
        avg = self._get_average_episode_tracker().get_average()
        self.log_dict["average_episode_length"] = avg.detach().cpu()

    ################ Curriculum #################

    def _get_average_episode_tracker(self):
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            raise RuntimeError("AverageEpisodeLengthTracker is not registered with the curriculum manager.")
        return tracker

    @property
    def average_episode_length(self) -> float:
        avg = self._get_average_episode_tracker().get_average()
        return float(avg.detach().cpu().item())

    # ------------------------------------------------------------------
    # Checkpoint helpers

    def get_checkpoint_state(self) -> dict[str, torch.Tensor | float]:
        state: dict[str, torch.Tensor | float] = {}
        state["average_episode_tracker"] = self._get_average_episode_tracker().state_dict()
        if hasattr(self, "reward_penalty_scale"):
            state["reward_penalty_scale"] = float(self.reward_penalty_scale)
        return state

    def load_checkpoint_state(self, state: dict[str, torch.Tensor | float] | None) -> None:
        if not state:
            return

        tracker_state = state.get("average_episode_tracker")
        if tracker_state is not None:
            tracker = self._get_average_episode_tracker()
            tracker.load_state_dict(tracker_state)
            tracker.suppress_next_update()

        penalty_state = state.get("reward_penalty_scale")
        if penalty_state is not None:
            if isinstance(penalty_state, torch.Tensor):
                self.reward_penalty_scale = float(penalty_state.item())
            else:
                self.reward_penalty_scale = float(penalty_state)

    def synchronize_curriculum_state(self, *, device: str, world_size: int) -> None:
        if world_size <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        tracker = self._get_average_episode_tracker()
        avg_tensor = tracker.get_average().clone().detach().to(device)
        torch.distributed.broadcast(avg_tensor, src=0)
        tracker.set_average(avg_tensor.to(self.device), suppress_update=False)

        if hasattr(self, "reward_penalty_scale"):
            penalty_tensor = torch.tensor(float(self.reward_penalty_scale), device=device, dtype=torch.float)
            torch.distributed.broadcast(penalty_tensor, src=0)
            self.reward_penalty_scale = float(penalty_tensor.item())

    def _push_robots(self, env_ids):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = self.randomization_manager.get_state("push_randomizer_state")
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 2:
            raise ValueError("Locomotion push velocity vector must have exactly 2 components.")

        rand = torch.rand(len(env_ids), 2, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        self.simulator.robot_root_states[env_ids, 7:9] = self.push_robot_vel_buf[env_ids]
        # Push impulses only take effect in the simulator once we write the mutated root state tensor back.
        self.simulator.set_actor_root_state_tensor_robots(env_ids, self.simulator.robot_root_states)
        self._max_push_vel = max_vel_tensor.clone()

    ################ ENV CALLBACKS #################

    def _reset_dofs(self, env_ids, target_state=None):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        If target_state is not None, reset to target_state

        Args:
            env_ids (List[int]): Environemnt ids
            target_state (Tensor): Target state
        """
        if target_state is not None:
            self.simulator.dof_pos[env_ids] = target_state[..., 0]
            self.simulator.dof_vel[env_ids] = target_state[..., 1]
        else:
            self.simulator.dof_pos[env_ids] = self.default_dof_pos[env_ids] * torch_rand_float(
                0.5, 1.5, (len(env_ids), self.num_dof), device=str(self.device)
            )
            self.simulator.dof_vel[env_ids] = 0.0

    def _reset_root_states(self, env_ids, target_root_states=None):
        """Resets ROOT states position and velocities of selected environmments
            if target_root_states is not None, reset to target_root_states
        Args:
            env_ids (List[int]): Environemnt ids
            target_root_states (Tensor): Target root states
        """
        if target_root_states is not None:
            self.simulator.robot_root_states[env_ids] = target_root_states
            self.simulator.robot_root_states[env_ids, :3] += self.terrain_manager.get_state(
                "locomotion_terrain"
            ).env_origins[env_ids]

        else:
            # base position
            self.simulator.robot_root_states[env_ids] = self.base_init_state
            self.simulator.robot_root_states[env_ids, :3] += self.terrain_manager.get_state(
                "locomotion_terrain"
            ).env_origins[env_ids]

            # Apply randomized XY offset if custom_origins
            if self.terrain_manager.get_state("locomotion_terrain").custom_origins:
                # Get spawn config from terrain term
                spawn_cfg = self.terrain_manager.cfg.terrain_term.spawn

                # Generate random XY offsets
                xy_offsets = torch_rand_float(
                    -spawn_cfg.xy_offset_range, spawn_cfg.xy_offset_range, (len(env_ids), 2), device=str(self.device)
                )

                if spawn_cfg.query_terrain_height:
                    # ACCURATE: Query terrain height at new XY position (slower, for rough terrain)
                    current_xy = self.simulator.robot_root_states[env_ids, :2]
                    new_xy = current_xy + xy_offsets

                    terrain_state = self.terrain_manager.get_state("locomotion_terrain")
                    terrain_heights = terrain_state.query_terrain_heights(
                        new_xy,
                        use_grid_sampling=spawn_cfg.use_grid_sampling,
                        grid_size=spawn_cfg.grid_size,
                        grid_spacing=spawn_cfg.grid_spacing,
                    )
                    robot_base_height = self.robot_config.init_state.pos[2]  # Robot base height above ground
                    new_z = terrain_heights + robot_base_height

                    # Write new XYZ position all at once
                    new_xyz = torch.cat([new_xy, new_z.unsqueeze(1)], dim=1)
                    self.simulator.robot_root_states[env_ids, :3] = new_xyz
                else:
                    # FAST: Original simple spawning - just apply XY offset, keep original Z (faster, for flat terrain)
                    self.simulator.robot_root_states[env_ids, :2] += xy_offsets

            # base velocities
            self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(
                -0.5, 0.5, (len(env_ids), 6), device=str(self.device)
            )  # [7:10]: lin vel, [10:13]: ang vel
