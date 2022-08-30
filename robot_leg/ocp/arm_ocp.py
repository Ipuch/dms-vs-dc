import warnings

from casadi import MX, vertcat

import biorbd_casadi as biorbd
import numpy as np
from scipy import interpolate
from bioptim import (
    OdeSolver,
    Node,
    OptimalControlProgram,
    ConstraintFcn,
    DynamicsFcn,
    ObjectiveFcn,
    QAndQDotBounds,
    QAndQDotAndQDDotBounds,
    ConstraintList,
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InitialGuessList,
    ControlType,
    InterpolationType,
    PhaseTransitionList,
    RigidBodyDynamics,
    NoisedInitialGuess,
    PenaltyNodeList,
)


class ArmOCP:
    def __init__(
        self,
        biorbd_model_path: str = None,
        n_shooting: int = 10,
        phase_time: float = 0.25,
        n_threads: int = 8,
        control_type: ControlType = ControlType.CONSTANT,
        ode_solver: OdeSolver = OdeSolver.RK4(),
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        seed: int = 0,
        use_sx: bool = False,
        start_point: np.array = np.array([0.2, -0.02, 0.1]),
        end_point: np.array = np.array([0.63, 0, -0.1]),
    ):
        self.biorbd_model_path = biorbd_model_path
        self.n_shooting = n_shooting
        self.phase_time = phase_time
        self.n_threads = n_threads
        self.control_type = control_type
        self.ode_solver = ode_solver
        self.rigidbody_dynamics = rigidbody_dynamics

        if biorbd_model_path is not None:
            self.biorbd_model = biorbd.Model(biorbd_model_path)
            self.n_shooting = n_shooting
            self.phase_time = phase_time

            self.n_q = self.biorbd_model.nbQ()
            self.n_qdot = self.biorbd_model.nbQdot()
            self.n_qddot = self.biorbd_model.nbQddot()
            self.n_qdddot = self.n_qddot
            self.n_tau = self.biorbd_model.nbGeneralizedTorque()

            self.tau_min, self.tau_init, self.tau_max = -100, 0, 50
            self.qddot_min, self.qddot_init, self.qddot_max = -100, 0, 100
            self.qdddot_min, self.qdddot_init, self.qdddot_max = -1000, 0, 1000

            self.start_point = start_point
            self.end_point = end_point

            self.dynamics = DynamicsList()
            self.constraints = ConstraintList()
            self.objective_functions = ObjectiveList()
            self.phase_transitions = PhaseTransitionList()
            self.x_bounds = BoundsList()
            self.u_bounds = BoundsList()
            self.initial_states = []
            self.x_init = InitialGuessList()
            self.u_init = InitialGuessList()

            self.control_type = control_type
            self.control_nodes = Node.ALL if self.control_type == ControlType.LINEAR_CONTINUOUS else Node.ALL_SHOOTING

            self._set_dynamics()
            self._set_constraints()
            self._set_objective_functions()

            self._set_boundary_conditions()
            self._set_initial_guesses()

            self.xn_init = InitialGuessList()
            self.un_init = InitialGuessList()

            self.xn_init.add(
                NoisedInitialGuess(
                    initial_guess=self.x_init[0],
                    bounds=self.x_bounds[0],
                    noise_magnitude=0.5,
                    n_shooting=self.n_shooting,
                    interpolation=InterpolationType.EACH_FRAME,
                    bound_push=0.1,
                    seed=seed,
                )
            )
            self.un_init.add(
                NoisedInitialGuess(
                    initial_guess=self.u_init[0],
                    bounds=self.u_bounds[0],
                    noise_magnitude=0.5,
                    n_shooting=self.n_shooting - 1,
                    bound_push=0.1,
                    seed=seed,
                )
            )

            self.ocp = OptimalControlProgram(
                self.biorbd_model,
                self.dynamics,
                self.n_shooting,
                self.phase_time,
                x_init=self.xn_init,
                x_bounds=self.x_bounds,
                u_init=self.un_init,
                u_bounds=self.u_bounds,
                objective_functions=self.objective_functions,
                constraints=self.constraints,
                n_threads=n_threads,
                control_type=self.control_type,
                ode_solver=ode_solver,
                use_sx=use_sx,
            )

    def _set_dynamics(self):
        self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN, rigidbody_dynamics=self.rigidbody_dynamics, phase=0)

    def _set_objective_functions(self):

        # --- Objective function --- #
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=0)
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=0, derivative=True)
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, weight=0.01)

        if (
            self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
            or self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
        ):
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, phase=0, key="qdddot", weight=1e-4)

    def _set_constraints(self):
        def last_segment_vertical(all_pn: PenaltyNodeList) -> MX:
            """
            The used-defined objective function

            Parameters
            ----------
            all_pn: PenaltyNodeList
                The penalty node elements

            Returns
            -------
            The z-axis in global frame of the last segment
            """

            rotation_matrix = all_pn.nlp.model.globalJCS(
                all_pn.nlp.states["q"].cx, all_pn.nlp.model.nbSegment() - 1
            ).to_mx()

            return vertcat(
                rotation_matrix[2, 0],
                rotation_matrix[2, 1],
                rotation_matrix[0, 2],
                rotation_matrix[0, 2],
                rotation_matrix[1, 2],
                (1 - rotation_matrix[2, 2]),
            )

        # --- Constraints --- #
        # Contact force in Z are positive
        node = Node.START
        self.constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, node=node, marker_index="marker_Leg1", phase=0)
        self.constraints.add(ConstraintFcn.TRACK_QDDOT, node=node, phase=0)

        node = Node.END
        self.constraints.add(
            ConstraintFcn.TRACK_MARKERS, node=node, target=self.end_point, marker_index="marker_Leg1", phase=0
        )
        self.constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, node=node, marker_index="marker_Leg1", phase=0)
        self.constraints.add(last_segment_vertical, node=node, phase=0, quadratic=True)

        # self.objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_QDDOT, target=np.zeros(self.n_q), node=Node.PENULTIMATE, phase=0, weight=10)
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_QDDOT, target=np.zeros(self.n_q), node=Node.END, phase=0, weight=10
        )
        # self.constraints.add(ConstraintFcn.TRACK_QDDOT, node=node, phase=0)

    def _set_boundary_conditions(self):
        self.x_bounds = BoundsList()
        self.x_bounds.add(
            bounds=QAndQDotAndQDDotBounds(self.biorbd_model)
            if self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
            or self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
            else QAndQDotBounds(self.biorbd_model)
        )
        nq = self.n_q
        self.x_bounds[0].max[: self.n_q, 0] = 0
        self.x_bounds[0].min[: self.n_q, 0] = 0
        self.x_bounds[0].max[self.n_q :, 0] = 0
        self.x_bounds[0].min[self.n_q :, 0] = 0
        self.x_bounds[0].max[self.n_q :, -1] = 0
        self.x_bounds[0].min[self.n_q :, -1] = 0

        if self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau + [self.qddot_min] * self.n_qddot,
                [self.tau_max] * self.n_tau + [self.qddot_max] * self.n_qddot,
            )
        elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau + [self.qddot_min] * self.n_qddot,
                [self.tau_max] * self.n_tau + [self.qddot_max] * self.n_qddot,
            )
        elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau + [self.qdddot_min] * self.n_qddot,
                [self.tau_max] * self.n_tau + [self.qdddot_max] * self.n_qddot,
            )
        elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau + [self.qdddot_min] * self.n_qddot,
                [self.tau_max] * self.n_tau + [self.qdddot_max] * self.n_qddot,
            )
        else:
            self.u_bounds.add([self.tau_min] * self.n_tau, [self.tau_max] * self.n_tau)

    def _set_initial_guesses(self):
        """
        Set initial guess for the optimization problem.
        """

        self._set_initial_states()
        self._set_initial_controls()

    def _set_initial_states(self, X0: np.array = None):
        X0 = np.zeros((self.n_q + self.n_qdot, self.n_shooting + 1)) if X0 is None else X0
        self.x_init.add(X0, interpolation=InterpolationType.EACH_FRAME)

    def _set_initial_controls(self, U0: np.array = None):
        if U0 is None:
            if self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
                self.u_init.add([self.tau_init] * self.n_tau + [self.qddot_init] * self.n_qddot)
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK:
                self.u_init.add([self.tau_init] * self.n_tau + [self.qdddot_init] * self.n_qdddot)
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK:
                self.u_init.add([self.tau_init] * self.n_tau + [self.qdddot_init] * self.n_qdddot)
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS:
                self.u_init.add([self.tau_init] * self.n_tau + [self.qddot_init] * self.n_qddot)
            else:
                self.u_init.add([self.tau_init] * self.n_tau)
        else:
            if U0.shape[1] != self.n_shooting:
                U0 = self._interpolate_initial_controls(U0)
            self.u_init.add(U0, interpolation=InterpolationType.EACH_FRAME)

    def _interpolate_initial_states(self, X0: np.array):
        print("interpolating initial states to match the number of shooting nodes")
        x = np.linspace(0, self.phase_time, X0.shape[1])
        y = X0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.phase_time, self.n_shooting + 1)
        y_new = f(x_new)  # use interpolation function returned by `interp1d`
        return y_new

    def _interpolate_initial_controls(self, U0: np.array):
        print("interpolating initial controls to match the number of shooting nodes")
        x = np.linspace(0, self.phase_time, U0.shape[1])
        y = U0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.phase_time, self.n_shooting)
        y_new = f(x_new)  # use interpolation function returned by `interp1d`
        return y_new
