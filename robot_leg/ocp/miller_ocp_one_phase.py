from enum import Enum

import biorbd_casadi as biorbd
import biorbd as brd
import numpy as np
from scipy import interpolate
from bioptim import (
    OdeSolver,
    Node,
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveFcn,
    ConstraintList,
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InitialGuessList,
    ControlType,
    Bounds,
    InterpolationType,
    PhaseTransitionList,
    BiMappingList,
    MultinodeConstraintList,
    RigidBodyDynamics,
    NoisedInitialGuess,
    IntegralApproximation,
)

# from custom_dynamics.root_explicit_qddot_joint import root_explicit_dynamic, custom_configure_root_explicit
# from custom_dynamics.root_implicit import root_implicit_dynamic, custom_configure_root_implicit
# from custom_dynamics.implicit_dynamics_tau_driven_qdddot import (
#     tau_implicit_qdddot_dynamic,
#     custom_configure_tau_driven_implicit,
# )
# from custom_dynamics.root_implicit_qddot import root_implicit_qdddot_dynamic, custom_configure_root_implicit_qdddot


class MillerDynamics(Enum):
    """
    Selection of dynamics to perform the miller ocp
    """

    EXPLICIT = "explicit"
    ROOT_EXPLICIT = "root_explicit"
    IMPLICIT = "implicit"
    ROOT_IMPLICIT = "root_implicit"
    IMPLICIT_TAU_DRIVEN_QDDDOT = "implicit_qdddot"
    ROOT_IMPLICIT_QDDDOT = "root_implicit_qdddot"


class MillerOcpOnePhase:
    """
    Class to generate the OCP for the miller acrobatic task for a 15-dof human model.

    Methods
    ----------
    _set_dynamics
        Set the dynamics of the OCP
    _set_objective
        Set the objective of the OCP
    _set_constraints
        Set the constraints of the OCP
    _set_bounds
        method to set the bounds of the OCP
    _set_initial_guess
        method to set the initial guess of the OCP
    _set_mapping
        method to set the mapping between variables of the model
    _print_bounds
        method to print the bounds of the states into the console
    """

    def __init__(
        self,
        biorbd_model_path: str = None,
        n_shooting: float = 125,
        phase_durations: float = 1.50187,  # actualized with results from https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4096894
        n_threads: int = 8,
        ode_solver: OdeSolver = OdeSolver.RK4(),
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        vertical_velocity_0: float = 8.30022867e00,  # actualized with results from https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4096894
        somersaults: float = 4 * np.pi,
        twists: float = 2 * np.pi,
        use_sx: bool = False,
        extra_obj: bool = False,
        initial_x: InitialGuessList = None,
        initial_u: InitialGuessList = None,
        seed: int = None,
    ):
        """
        Parameters
        ----------
        biorbd_model_path : str
            path to the biorbd model
        n_shooting : tuple
            number of shooting points for each phase
        phase_durations : tuple
            duration of each phase
        n_threads : int
            number of threads to use for the solver
        ode_solver : OdeSolver
            type of ordinary differential equation solver to use
        rigidbody_dynamics : RigidBodyDynamics
            type of dynamics to use
        vertical_velocity_0 : float
            initial vertical velocity of the model to execute the Miller task
        somersaults : float
            number of somersaults to execute
        twists : float
            number of twists to execute
        use_sx : bool
            use SX for the dynamics
        extra_obj : bool
            use extra objective to the extra controls of implicit dynamics (algebraic states)
        initial_x : InitialGuessList
            initial guess for the states
        initial_u : InitialGuessList
            initial guess for the controls
        seed : int
            seed for the random generator
        """
        self.biorbd_model_path = biorbd_model_path
        self.extra_obj = extra_obj
        self.n_shooting = n_shooting
        self.n_phases = 1

        self.somersaults = somersaults
        self.twists = twists

        self.x = None
        self.u = None

        self.phase_durations = phase_durations
        self.phase_time = phase_durations

        self.duration = np.sum(self.phase_durations)
        self.phase_proportions = 0.8966564714409299

        self.velocity_x = 0
        self.velocity_y = 0
        self.vertical_velocity_0 = vertical_velocity_0
        self.somersault_rate_0 = self.somersaults / (self.duration + 0.193125)

        self.n_threads = n_threads
        self.ode_solver = ode_solver

        if biorbd_model_path is not None:

            self.biorbd_model = (biorbd.Model(biorbd_model_path),)
            self.rigidbody_dynamics = rigidbody_dynamics

            self.n_q = self.biorbd_model[0].nbQ()
            self.n_qdot = self.biorbd_model[0].nbQdot()
            self.nb_root = self.biorbd_model[0].nbRoot()

            if (
                self.rigidbody_dynamics == MillerDynamics.IMPLICIT
                or self.rigidbody_dynamics == MillerDynamics.ROOT_IMPLICIT
                or self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
                or self.rigidbody_dynamics == MillerDynamics.ROOT_IMPLICIT_QDDDOT
            ):
                self.n_qddot = self.biorbd_model[0].nbQddot()
            elif (
                self.rigidbody_dynamics == RigidBodyDynamics.ODE
                or self.rigidbody_dynamics == MillerDynamics.ROOT_EXPLICIT
            ):
                self.n_qddot = self.biorbd_model[0].nbQddot() - self.biorbd_model[0].nbRoot()

            self.n_tau = self.biorbd_model[0].nbGeneralizedTorque() - self.biorbd_model[0].nbRoot()

            if (
                self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
                or self.rigidbody_dynamics == MillerDynamics.ROOT_IMPLICIT_QDDDOT
            ):
                self.n_qdddot = self.biorbd_model[0].nbQddot()

            self.tau_min, self.tau_init, self.tau_max = -100, 0, 100
            self.tau_hips_min, self.tau_hips_init, self.tau_hips_max = (
                -300,
                0,
                300,
            )  # hips and torso

            self.high_torque_idx = [
                6 - self.nb_root,
                7 - self.nb_root,
                12 - self.nb_root,
                13 - self.nb_root,
            ]

            self.qddot_min, self.qddot_init, self.qddot_max = -1000, 0, 1000

            if (
                self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
                or self.rigidbody_dynamics == MillerDynamics.ROOT_IMPLICIT_QDDDOT
            ):
                self.qdddot_min, self.qdddot_init, self.qdddot_max = (
                    -1000 * 10,
                    0,
                    1000 * 10,
                )

            self.velocity_max = 100  # qdot
            self.velocity_max_phase_transition = 10  # qdot hips, thorax in phase 2

            self.random_scale = 0.02  # relative to the maximal bounds of the states or controls
            self.random_scale_qdot = 0.02
            self.random_scale_qddot = 0.02
            self.random_scale_tau = 0.02

            self.dynamics = DynamicsList()
            self.constraints = ConstraintList()
            self.objective_functions = ObjectiveList()
            self.phase_transitions = PhaseTransitionList()
            self.multinode_constraints = MultinodeConstraintList()
            self.x_bounds = BoundsList()
            self.u_bounds = BoundsList()
            self.initial_states = []
            self.x_init = InitialGuessList() if initial_x is None else initial_x
            self.u_init = InitialGuessList() if initial_u is None else initial_u
            self.mapping = BiMappingList()

            self._set_boundary_conditions()

            np.random.seed(seed)
            if initial_x is None:
                self._set_initial_guesses()  # noise is into the initial guess
            if initial_u is None:
                self._set_initial_controls()  # noise is into the initial guess

            self._set_dynamics()
            self._set_objective_functions()

            self._set_mapping()

            self.ocp = OptimalControlProgram(
                self.biorbd_model,
                self.dynamics,
                n_shooting=self.n_shooting,
                phase_time=self.phase_durations,
                x_init=self.x_init,
                x_bounds=self.x_bounds,
                u_init=self.u_init,
                u_bounds=self.u_bounds,
                objective_functions=self.objective_functions,
                phase_transitions=self.phase_transitions,
                multinode_constraints=self.multinode_constraints,
                n_threads=n_threads,
                variable_mappings=self.mapping,
                control_type=ControlType.CONSTANT,
                ode_solver=ode_solver,
                use_sx=use_sx,
            )

    def _set_dynamics(self):
        """
        Set the dynamics of the optimal control problem
        """

        if self.rigidbody_dynamics == RigidBodyDynamics.ODE:
            self.dynamics.add(
                DynamicsFcn.TORQUE_DRIVEN,
                with_contact=False,
                rigidbody_dynamics=RigidBodyDynamics.ODE,
            )
        else:
            raise ValueError("This dynamics has not been implemented")

    def _set_objective_functions(self):
        """
        Set the multi-objective functions for each phase with specific weights
        """

        # --- Objective function --- #
        w_qdot = 1
        w_penalty = 1
        w_penalty_foot = 10
        w_penalty_core = 10
        w_torque = 100
        # integral_approximation = IntegralApproximation.TRAPEZOIDAL
        i = 0
        ## TAU ##
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            weight=w_torque,
            phase=i,
            quadratic=True,
        )
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            weight=w_torque,
            derivative=True,
            phase=i,
            quadratic=True,
            # integration_rule=integral_approximation
        )
        ## ANGULAR MOMENTUM ##
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_ANGULAR_MOMENTUM,
            weight=50,
            quadratic=False,
            index=[0],
            phase=i,
            node=Node.START,
        )
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_ANGULAR_MOMENTUM,
            weight=50,
            quadratic=True,
            index=[1, 2],
            phase=i,
            node=Node.START,
        )

        ## CORE DOF ##
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            index=(6, 7, 8, 13, 14),
            key="q",
            weight=w_penalty_core,
            phase=i,
            quadratic=True,
            # integration_rule=integral_approximation
        )  # core DoFs

        ## QDOT DERIVATIVE##
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            derivative=True,
            key="qdot",
            index=(6, 7, 8, 9, 10, 11, 12, 13, 14),
            weight=w_qdot,
            phase=i,
            quadratic=True,
            # integration_rule=integral_approximation
        )  # Regularization

        ## MARKERS POSITION ##
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_MARKERS,
            derivative=True,
            reference_jcs=0,
            marker_index=6,
            weight=w_penalty,
            phase=i,
            node=Node.ALL_SHOOTING,
            quadratic=True,
        )  # Right hand trajectory
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_MARKERS,
            derivative=True,
            reference_jcs=0,
            marker_index=11,
            weight=w_penalty,
            phase=i,
            node=Node.ALL_SHOOTING,
            quadratic=True,
        )  # Left hand trajectory
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_MARKERS,  # Lagrange
            node=Node.ALL_SHOOTING,
            derivative=True,
            reference_jcs=0,
            marker_index=16,
            weight=w_penalty_foot,
            phase=i,
            quadratic=True,
        )  # feet trajectory

        ## SLACKED TIME ##
        slack_duration = 0.05
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME,
            min_bound=self.phase_durations - slack_duration,
            max_bound=self.phase_durations + slack_duration,
            phase=0,
            weight=1e-6,
        )

    def _set_initial_guesses(self):
        """
        Set the initial guess for the optimal control problem (states and controls)
        """
        # --- Initial guess --- #
        total_n_shooting = self.n_shooting + 1
        # Initialize state vector
        # if self.x is None:
        self.x = np.zeros((self.n_q + self.n_qdot, total_n_shooting))

        # determine v such that final z == 0
        v0 = 1 / 2 * 9.81 * self.duration  #
        # time vector
        data_point = np.linspace(0, self.duration, total_n_shooting)
        # parabolic trajectory on Z
        self.x[2, :] = v0 * data_point + -9.81 / 2 * data_point**2
        # Somersaults
        self.x[3, :] = np.linspace(0, self.phase_proportions * self.somersaults, self.n_shooting + 1)
        # Twists
        self.x[5, :] = np.linspace(0, self.twists, self.n_shooting + 1)

        # Handle second DoF of arms with Noise.
        thorax_slice = range(6, 9)
        arm_slice = 10
        arm_slice2 = 12
        rest_slice = range(13, 15)

        self.x[thorax_slice, :] = np.random.random((3, total_n_shooting)) * np.pi / 12 - np.pi / 24
        self.x[arm_slice, :] = np.random.random((1, total_n_shooting)) * np.pi / 2 - (np.pi - np.pi / 4)
        self.x[arm_slice2, :] = np.random.random((1, total_n_shooting)) * np.pi / 2 + np.pi / 4
        self.x[rest_slice, :] = np.random.random((2, total_n_shooting)) * np.pi / 12 - np.pi / 24

        # velocity on Y
        self.x[self.n_q + 0, :] = self.velocity_x
        self.x[self.n_q + 1, :] = self.velocity_y
        self.x[self.n_q + 2, :] = self.vertical_velocity_0 - 9.81 * data_point
        # Somersaults rate
        self.x[self.n_q + 3, :] = self.somersault_rate_0
        # Twists rate
        self.x[self.n_q + 5, :] = self.twists / self.duration

        # random for other velocities
        self.x[self.n_q + 6 :, :] = (
            (np.random.random((self.n_qdot - self.nb_root, total_n_shooting)) * 2 - 1)
            * self.velocity_max
            * self.random_scale_qdot
        )

        self._set_initial_states(self.x)

    def _set_initial_states(self, X0: np.array = None):
        """
        Set the initial states of the optimal control problem.
        """
        X0 = np.zeros((self.n_q + self.n_qdot, self.n_shooting + 1)) if X0 is None else X0
        self.x_init.add(X0, interpolation=InterpolationType.EACH_FRAME)

    def _set_initial_controls(self, U0: np.array = None):
        if U0 is None and self.u is None:
            n_shooting = self.n_shooting
            tau_J_random = np.random.random((self.n_tau, n_shooting)) * 2 - 1

            tau_max = self.tau_max * np.ones(self.n_tau)
            tau_max[self.high_torque_idx] = self.tau_hips_max
            tau_J_random = tau_J_random * tau_max[:, np.newaxis] * self.random_scale_tau

            qddot_J_random = (
                (np.random.random((self.n_tau, n_shooting)) * 2 - 1) * self.qddot_max * self.random_scale_qddot
            )
            qddot_B_random = (
                (np.random.random((self.nb_root, n_shooting)) * 2 - 1) * self.qddot_max * self.random_scale_qddot
            )

            if self.rigidbody_dynamics == RigidBodyDynamics.ODE:
                self.u_init.add(tau_J_random, interpolation=InterpolationType.EACH_FRAME)
            elif self.rigidbody_dynamics == MillerDynamics.ROOT_EXPLICIT:
                self.u_init.add(qddot_J_random, interpolation=InterpolationType.EACH_FRAME)
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
                u = np.vstack((tau_J_random, qddot_B_random, qddot_J_random))
                self.u_init.add(u, interpolation=InterpolationType.EACH_FRAME)
            elif self.rigidbody_dynamics == MillerDynamics.ROOT_IMPLICIT:
                u = np.vstack((qddot_B_random, qddot_J_random))
                self.u_init.add(u, interpolation=InterpolationType.EACH_FRAME)
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK:
                u = np.vstack((tau_J_random, qddot_B_random * 10, qddot_J_random * 10))
                self.u_init.add(u, interpolation=InterpolationType.EACH_FRAME)
            elif self.rigidbody_dynamics == MillerDynamics.ROOT_IMPLICIT_QDDDOT:
                u = np.vstack((qddot_B_random * 10, qddot_J_random * 10))
                self.u_init.add(u, interpolation=InterpolationType.EACH_FRAME)
            else:
                raise ValueError("This dynamics has not been implemented")
        elif self.u is not None:
            self.u_init.add(self.u[:, :-1], interpolation=InterpolationType.EACH_FRAME)
        else:
            if U0.shape[1] != self.n_shooting:
                U0 = self._interpolate_initial_controls(U0)

                shooting = 0
                self.u_init.add(
                    U0[:, shooting : shooting + self.n_shooting],
                    interpolation=InterpolationType.EACH_FRAME,
                )
                shooting += self.n_shooting

    def _set_boundary_conditions(self):
        """
        Set the boundary conditions for controls and states for each phase.
        """
        self.x_bounds = BoundsList()

        tilt_bound = np.pi / 4
        tilt_final_bound = np.pi / 12  # 15 degrees

        initial_arm_elevation = 2.8
        arm_rotation_z_upp = np.pi / 2
        arm_rotation_z_low = 1
        arm_elevation_y_low = 0.01
        arm_elevation_y_upp = np.pi - 0.01
        thorax_hips_xyz = np.pi / 6
        self.thorax_hips_xyz = thorax_hips_xyz
        arm_rotation_y_final = 2.4

        slack_initial_vertical_velocity = 2
        slack_initial_somersault_rate = 3
        slack_initial_translation_velocities = 1

        # end phase 0
        slack_somersault = 30 * 3.14 / 180
        slack_twist = 30 * 3.14 / 180

        slack_final_somersault = np.pi / 24  # 7.5 degrees
        self.slack_final_somersault = slack_final_somersault
        slack_final_twist = np.pi / 24  # 7.5 degrees
        slack_final_dofs = np.pi / 24  # 7.5 degrees

        x_min = np.zeros((self.n_q + self.n_qdot, 3))
        x_max = np.zeros((self.n_q + self.n_qdot, 3))

        x_min[: self.n_q, 0] = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -initial_arm_elevation,
            0,
            initial_arm_elevation,
            0,
            0,
        ]
        x_min[self.n_q :, 0] = [
            self.velocity_x - slack_initial_translation_velocities,
            self.velocity_y - slack_initial_translation_velocities,
            self.vertical_velocity_0 - slack_initial_vertical_velocity,
            self.somersault_rate_0 - slack_initial_somersault_rate,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        x_max[: self.n_q, 0] = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -initial_arm_elevation,
            0,
            initial_arm_elevation,
            0,
            0,
        ]
        x_max[self.n_q :, 0] = [
            self.velocity_x + slack_initial_translation_velocities,
            self.velocity_y + slack_initial_translation_velocities,
            self.vertical_velocity_0 + slack_initial_vertical_velocity,
            self.somersault_rate_0 + slack_initial_somersault_rate,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        x_min[: self.n_q, 1] = [
            -3,
            -3,
            -0.001,
            -0.001,
            -tilt_bound,
            -0.001,
            -thorax_hips_xyz,
            -thorax_hips_xyz,
            -thorax_hips_xyz,
            -arm_rotation_z_low,
            -arm_elevation_y_upp,
            -arm_rotation_z_upp,
            arm_elevation_y_low,
            -thorax_hips_xyz,
            -thorax_hips_xyz,
        ]
        x_min[self.n_q :, 1] = -self.velocity_max

        x_max[: self.n_q, 1] = [
            3,
            3,
            10,
            self.somersaults + slack_somersault,
            tilt_bound,
            self.twists + slack_twist,
            thorax_hips_xyz,
            thorax_hips_xyz,
            thorax_hips_xyz,
            arm_rotation_z_upp,
            -arm_elevation_y_low,
            arm_rotation_z_low,
            arm_elevation_y_upp,
            thorax_hips_xyz,
            thorax_hips_xyz,
        ]
        x_max[self.n_q :, 1] = +self.velocity_max

        x_min[: self.n_q, 2] = [
            -0.01,
            -0.1,
            1.45,
            self.somersaults - 2 * np.pi / 4 - slack_somersault,
            -tilt_final_bound,
            self.twists - slack_twist,
            -slack_final_dofs,
            -slack_final_dofs,
            -slack_final_dofs,
            -arm_rotation_z_low,
            -0.2,
            -arm_rotation_z_upp,
            arm_elevation_y_low,
            thorax_hips_xyz - slack_final_dofs,
            -slack_final_dofs,
        ]

        x_max[: self.n_q, 2] = [
            0.01,
            0.1,
            1.50,
            self.somersaults - 2 * np.pi / 4 + slack_somersault,
            tilt_final_bound,
            self.twists + slack_twist,
            slack_final_dofs,
            slack_final_dofs,
            slack_final_dofs,
            arm_rotation_z_upp,
            -arm_elevation_y_low,
            arm_rotation_z_low,
            0.2,
            thorax_hips_xyz,
            slack_final_dofs,
        ]

        x_min[self.n_q :, 2] = -20
        x_max[self.n_q :, 2] = 20

        self.x_bounds.add(
            bounds=Bounds(
                x_min,
                x_max,
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )
        )

        if self.rigidbody_dynamics == RigidBodyDynamics.ODE:
            self.u_bounds.add([self.tau_min] * self.n_tau, [self.tau_max] * self.n_tau)
            self.u_bounds[0].min[self.high_torque_idx, :] = self.tau_hips_min
            self.u_bounds[0].max[self.high_torque_idx, :] = self.tau_hips_max
        # elif self.rigidbody_dynamics == MillerDynamics.ROOT_EXPLICIT:
        #     self.u_bounds.add([self.qddot_min] * self.n_qddot, [self.qddot_max] * self.n_qddot)
        # elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        #     self.u_bounds.add(
        #         [self.tau_min] * self.n_tau + [self.qddot_min] * self.n_qddot,
        #         [self.tau_max] * self.n_tau + [self.qddot_max] * self.n_qddot,
        #     )
        #     self.u_bounds[0].min[self.high_torque_idx, :] = self.tau_hips_min
        #     self.u_bounds[0].max[self.high_torque_idx, :] = self.tau_hips_max
        # elif self.rigidbody_dynamics == MillerDynamics.ROOT_IMPLICIT:
        #     self.u_bounds.add([self.qddot_min] * self.n_qddot, [self.qddot_max] * self.n_qddot)
        # elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK:
        #     self.u_bounds.add(
        #         [self.tau_min] * self.n_tau + [self.qdddot_min] * self.n_qdddot,
        #         [self.tau_max] * self.n_tau + [self.qdddot_max] * self.n_qdddot,
        #     )
        #     self.u_bounds[0].min[self.high_torque_idx, :] = self.tau_hips_min
        #     self.u_bounds[0].max[self.high_torque_idx, :] = self.tau_hips_max
        # elif self.rigidbody_dynamics == MillerDynamics.ROOT_IMPLICIT_QDDDOT:
        #     self.u_bounds.add(
        #         [self.qdddot_min] * self.n_qdddot,
        #         [self.qdddot_max] * self.n_qdddot,
        #     )
        # else:
        #     raise ValueError("This dynamics has not been implemented")

    def _interpolate_initial_states(self, X0: np.array):
        """
        Interpolate the initial states to match the number of shooting nodes
        """
        print("interpolating initial states to match the number of shooting nodes")
        x = np.linspace(0, self.phase_time, X0.shape[1])
        y = X0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.phase_time, np.sum(self.n_shooting) + len(self.n_shooting))
        y_new = f(x_new)  # use interpolation function returned by `interp1d`
        return y_new

    def _interpolate_initial_controls(self, U0: np.array):
        """
        Interpolate the initial controls to match the number of shooting nodes
        """
        print("interpolating initial controls to match the number of shooting nodes")
        x = np.linspace(0, self.phase_time, U0.shape[1])
        y = U0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.phase_time, self.n_shooting)
        y_new = f(x_new)  # use interpolation function returned by `interp1d`
        return y_new

    def _set_mapping(self):
        """
        Set the mapping between the states and controls of the model
        """
        if self.rigidbody_dynamics == RigidBodyDynamics.ODE:
            self.mapping.add(
                "tau",
                [None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                [6, 7, 8, 9, 10, 11, 12, 13, 14],
            )
        elif self.rigidbody_dynamics == MillerDynamics.ROOT_EXPLICIT:
            print("no bimapping")
        elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            self.mapping.add(
                "tau",
                [None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                [6, 7, 8, 9, 10, 11, 12, 13, 14],
            )
        elif self.rigidbody_dynamics == MillerDynamics.ROOT_IMPLICIT:
            pass
        elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK:
            self.mapping.add(
                "tau",
                [None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                [6, 7, 8, 9, 10, 11, 12, 13, 14],
            )
        elif self.rigidbody_dynamics == MillerDynamics.ROOT_IMPLICIT_QDDDOT:
            pass
        else:
            raise ValueError("This dynamics has not been implemented")
