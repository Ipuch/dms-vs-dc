import warnings
from typing import Union

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
    BiorbdModel,
)


class LegOCP:
    def __init__(
        self,
        biorbd_model_path: Union[str, tuple] = None,
        n_shooting: Union[int, tuple] = 10,
        phase_time: Union[float, tuple] = 0.25,
        n_threads: int = 8,
        control_type: ControlType = ControlType.CONSTANT,
        ode_solver: OdeSolver = OdeSolver.RK4(),
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        seed: int = 0,
        use_sx: bool = False,
        start_point: np.array = np.array([0.22, 0.02, 0.03]),
        end_point: np.array = np.array([0.22, 0.021, -0.05]),
    ):
        self.biorbd_model_path = biorbd_model_path
        # self.n_shooting = [n_shooting] if isinstance(n_shooting, int) else n_shooting
        self.phase_time = phase_time
        self.n_threads = n_threads
        self.control_type = control_type
        self.ode_solver = ode_solver
        self.rigidbody_dynamics = rigidbody_dynamics

        if biorbd_model_path is not None:

            self.n_shooting = [n_shooting] if isinstance(n_shooting, int) else n_shooting
            self.phase_time = phase_time
            self.n_phase = 1 if isinstance(self.n_shooting, int) else len(self.n_shooting)
            self.biorbd_model = [BiorbdModel(biorbd_model_path) for _ in range(self.n_phase)]

            self.n_q = self.biorbd_model[0].nb_q
            self.n_qdot = self.biorbd_model[0].nb_qdot
            self.n_qddot = self.biorbd_model[0].nb_qddot
            self.n_qdddot = self.n_qddot
            self.n_tau = self.biorbd_model[0].nb_tau

            self.tau_min, self.tau_init, self.tau_max = -0.5, 0, 0.5
            self.qddot_min, self.qddot_init, self.qddot_max = -100, 0, 100
            self.qdddot_min, self.qdddot_init, self.qdddot_max = -10000, 0, 1000

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

            for i in range(self.n_phase):
                self.xn_init.add(
                    NoisedInitialGuess(
                        initial_guess=self.x_init[i],
                        bounds=self.x_bounds[i],
                        noise_magnitude=1,
                        # noise_magnitude_bounds=0,
                        n_shooting=self.n_shooting[i] + 1,
                        bound_push=0.1,
                        seed=seed,
                    )
                )

                self.un_init.add(
                    NoisedInitialGuess(
                        initial_guess=self.u_init[i],
                        bounds=self.u_bounds[i],
                        noise_magnitude=0.2,
                        # noise_magnitude=0,
                        n_shooting=self.n_shooting[i],
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
        for i in range(self.n_phase):
            self.dynamics.add(
                DynamicsFcn.TORQUE_DRIVEN,
                rigidbody_dynamics=self.rigidbody_dynamics,
                phase=i,
            )

    def _set_objective_functions(self):
        # --- Objective function --- #
        for i in range(self.n_phase):
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=i)
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE,
                key="qdot",
                weight=1e-6,
                phase=i,
                derivative=True,
            )

            if (
                self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
                or self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
            ):
                self.objective_functions.add(
                    ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                    phase=i,
                    key="qdddot",
                    weight=1e-4,
                )
            if i == 1:
                self.objective_functions.add(
                    ObjectiveFcn.Mayer.MINIMIZE_QDDOT,
                    phase=i,
                    weight=2000,
                    node=Node.START,
                )

    def _set_constraints(self):
        # --- Constraints --- #
        # Contact force in Z are positive
        for i in range(self.n_phase):
            start_point = self.start_point if i == 0 else self.end_point
            end_point = self.end_point if i == 0 else self.start_point

            node = Node.START
            self.constraints.add(
                ConstraintFcn.TRACK_MARKERS,
                node=node,
                target=start_point,
                marker_index="marker_Leg1",
                phase=i,
            )
            self.constraints.add(
                ConstraintFcn.TRACK_MARKERS_VELOCITY,
                node=node,
                marker_index="marker_Leg1",
                phase=i,
            )

            node = Node.END
            self.constraints.add(
                ConstraintFcn.TRACK_MARKERS,
                node=node,
                target=end_point,
                marker_index="marker_Leg1",
                phase=i,
            )
            self.constraints.add(
                ConstraintFcn.TRACK_MARKERS_VELOCITY,
                node=node,
                marker_index="marker_Leg1",
                phase=i,
            )

    def _set_boundary_conditions(self):

        for i in range(self.n_phase):
            self.x_bounds.add(
                bounds=QAndQDotAndQDDotBounds(self.biorbd_model[i])
                if self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
                or self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
                else QAndQDotBounds(self.biorbd_model[i])
            )
            self.x_bounds[i].max[self.n_q, 1] = 3
            self.x_bounds[i].min[self.n_q, 1] = -3
            self.x_bounds[i].max[self.n_q+1, 1] = 10
            self.x_bounds[i].min[self.n_q+1, 1] = -3
            self.x_bounds[i].max[self.n_q+2, 1] = 16
            self.x_bounds[i].min[self.n_q+2, 1] = -16
            self.x_bounds[i].max[self.n_q : self.n_q + self.n_qdot, 0] = 0
            self.x_bounds[i].min[self.n_q : self.n_q + self.n_qdot, 0] = 0
            self.x_bounds[i].max[self.n_q : self.n_q + self.n_qdot, -1] = 0
            self.x_bounds[i].min[self.n_q : self.n_q + self.n_qdot, -1] = 0
            nq = self.n_q

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
        for i in range(self.n_phase):
            self._set_initial_states(phase=i)
            self._set_initial_controls(phase=i)

    def _set_initial_states(self, X0: np.array = None, phase: int = 0):
        X0 = np.zeros((self.n_q + self.n_qdot, self.n_shooting[phase] + 1)) if X0 is None else X0
        self.x_init.add(X0, interpolation=InterpolationType.EACH_FRAME)

    def _set_initial_controls(self, U0: np.array = None, phase: int = None):
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
            if U0.shape[1] != self.n_shooting[phase]:
                U0 = self._interpolate_initial_controls(U0, phase=phase)
            self.u_init.add(U0, interpolation=InterpolationType.EACH_FRAME)

    def _interpolate_initial_states(self, X0: np.array, phase=None):
        print("interpolating initial states to match the number of shooting nodes")
        x = np.linspace(0, self.phase_time, X0.shape[1])
        y = X0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.phase_time, self.n_shooting[phase] + 1)
        y_new = f(x_new)  # use interpolation function returned by `interp1d`
        return y_new

    def _interpolate_initial_controls(self, U0: np.array):
        print("interpolating initial controls to match the number of shooting nodes")
        x = np.linspace(0, self.phase_time, U0.shape[1])
        y = U0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.phase_time, self.n_shooting[phase])
        y_new = f(x_new)  # use interpolation function returned by `interp1d`
        return y_new
