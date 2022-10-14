from typing import Union

import numpy as np
from scipy import interpolate, optimize

import biorbd_casadi as biorbd
import biorbd as biorbd_eigen

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
    PhaseTransitionFcn,
    RigidBodyDynamics,
    MultinodeConstraintFcn,
    MultinodeConstraintList,
    NoisedInitialGuess,
)


class HumanoidOcpMultiPhase:
    """
    This class is used to create an optimal control program for a 2d humanoid stick model

    Attributes
    ----------
    biorbd_model_path: Union[str, tuple[str]]
        The biorbd models of the stickman
    n_shooting:  Union[int, list[int]]
        The number of shooting points
    phase_time:  Union[float, list[float]]
        The time of each phase
    n_threads: int
        The number of threads to use
    control_type: ControlType
        The type of control to use
    ode_solver: OdeSolver
        The ode solver to use
    rigidbody_dynamics: RigidBodyDynamics
        The type of rigidbody dynamics to use


    """

    def __init__(
        self,
        biorbd_model_path: Union[str, tuple[str]] = None,
        n_shooting: Union[int, list[int]] = 10,
        phase_time: Union[float, list[float]] = 0.3,
        n_threads: int = 8,
        control_type: ControlType = ControlType.CONSTANT,
        ode_solver: OdeSolver = OdeSolver.COLLOCATION(),
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        step_length: float = 0.8,
        right_foot_location: np.array = np.zeros(3),
        nb_phases: int = 1,
        seed: int = None,
        use_sx: bool = False,
    ):
        self.biorbd_model_path = biorbd_model_path
        self.n_shooting = n_shooting
        self.phase_time = phase_time
        self.n_threads = n_threads
        self.control_type = control_type
        self.ode_solver = ode_solver
        self.rigidbody_dynamics = rigidbody_dynamics

        if biorbd_model_path is not None:
            if nb_phases == 1:
                self.biorbd_model = (
                    (biorbd.Model(biorbd_model_path),)
                    if isinstance(biorbd_model_path, str)
                    else (biorbd.Model(biorbd_model_path[0]),)
                )
                self.n_shooting = (n_shooting,) if isinstance(n_shooting, int) else (n_shooting[0],)
                self.phase_time = (phase_time,) if isinstance(phase_time, float) else (phase_time[0],)
            else:
                self.biorbd_model = biorbd.Model(biorbd_model_path[0]), biorbd.Model(biorbd_model_path[1])
                self.n_shooting = n_shooting, n_shooting if isinstance(n_shooting, int) else n_shooting
                self.phase_time = phase_time, phase_time if isinstance(phase_time, float) else phase_time

            self._set_head()
            self._set_knee()
            self._set_shoulder()

            self.n_q = self.biorbd_model[0].nbQ()
            self.n_qdot = self.biorbd_model[0].nbQdot()
            self.n_qddot = self.biorbd_model[0].nbQddot()
            self.n_qdddot = self.n_qddot
            self.n_tau = self.biorbd_model[0].nbGeneralizedTorque()

            self.tau_min, self.tau_init, self.tau_max = -500, 0, 500
            self.qddot_min, self.qddot_init, self.qddot_max = -1000, 0, 1000
            self.qdddot_min, self.qdddot_init, self.qdddot_max = -10000, 0, 10000
            self.fext_min, self.fext_init, self.fext_max = -10000, 0, 10000

            self.right_foot_location = right_foot_location
            self.step_length = step_length
            self.initial_left_foot_location = right_foot_location - np.array([0, step_length / 2, 0])
            self.final_left_foot_location = right_foot_location + np.array([0, step_length / 2, 0])
            self.nb_phases = nb_phases
            if nb_phases == 2:
                self.left_foot_location = self.final_left_foot_location
                self.initial_right_foot_location = right_foot_location
                self.final_right_foot_location = right_foot_location + np.array([0, step_length, 0])

            self.dynamics = DynamicsList()
            self.constraints = ConstraintList()
            self.objective_functions = ObjectiveList()
            self.multinode_constraints = MultinodeConstraintList()
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
            self._set_phase_transition()

            self._set_boundary_conditions()
            self._set_initial_guesses()

            if seed is not None:
                self.xn_init = InitialGuessList()
                self.un_init = InitialGuessList()

                self.xn_init.add(
                    NoisedInitialGuess(
                        initial_guess=self.x_init[0],
                        bounds=self.x_bounds[0],
                        noise_magnitude=1,
                        n_shooting=self.n_shooting[0],
                        bound_push=0.1,
                        seed=seed,
                    )
                )
                self.un_init.add(
                    NoisedInitialGuess(
                        initial_guess=self.u_init[0],
                        bounds=self.u_bounds[0],
                        noise_magnitude=1,
                        n_shooting=self.n_shooting[0] - 1,
                        bound_push=0.1,
                        seed=seed,
                    )
                )

            self.ocp = OptimalControlProgram(
                self.biorbd_model,
                self.dynamics,
                self.n_shooting,
                self.phase_time,
                x_init=self.x_init if seed is None else self.xn_init,
                x_bounds=self.x_bounds,
                u_init=self.u_init if seed is None else self.un_init,
                u_bounds=self.u_bounds,
                objective_functions=self.objective_functions,
                constraints=self.constraints,
                phase_transitions=self.phase_transitions,
                multinode_constraints=self.multinode_constraints,
                n_threads=n_threads,
                control_type=self.control_type,
                ode_solver=ode_solver,
                use_sx=use_sx,
            )

    def _set_head(self):
        self.has_head = False
        for i in range(self.biorbd_model[0].nbSegment()):
            seg = self.biorbd_model[0].segment(i)
            if seg.name().to_string() == "Head":
                self.has_head = True
                break

    def _set_knee(self):
        self.has_knee = False
        for i in range(self.biorbd_model[0].nbSegment()):
            seg = self.biorbd_model[0].segment(i)
            if seg.name().to_string() == "RShank":
                self.has_knee = True
                break

    def _set_shoulder(self):
        self.has_shoulder = False
        for i in range(self.biorbd_model[0].nbSegment()):
            seg = self.biorbd_model[0].segment(i)
            if seg.name().to_string() == "RArm":
                self.has_shoulder = True
                break

    def _set_dynamics(self):
        """
        Set the dynamics of the optimal control problem

        """
        self.dynamics.add(
            DynamicsFcn.TORQUE_DRIVEN, rigidbody_dynamics=self.rigidbody_dynamics, with_contact=True, phase=0
        )
        if self.nb_phases == 2:
            self.dynamics.add(
                DynamicsFcn.TORQUE_DRIVEN, rigidbody_dynamics=self.rigidbody_dynamics, with_contact=True, phase=1
            )

    def _set_objective_functions(self):
        # --- Objective function --- #
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=0)
        if self.nb_phases == 2:
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=1)

        idx_stability = [0, 1, 2, 3] if self.has_head else [0, 1, 2]

        # torso stability
        for i in range(self.nb_phases):
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_QDDOT, index=idx_stability, weight=0.01, phase=i
            )

            # head stability
            if self.has_head:
                self.objective_functions.add(
                    ObjectiveFcn.Lagrange.MINIMIZE_QDDOT, derivative=True, index=3, weight=0.01, phase=i
                )
                self.objective_functions.add(
                    ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=3, weight=0.01, phase=i
                )

            # keep velocity CoM around 1.5 m/s
            com_velocity = 1.3  # old 1.5
            self.objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY,
                index=1,
                target=com_velocity,
                node=Node.START,
                weight=1000,
                phase=i,
            )
            self.objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY,
                index=1,
                target=com_velocity,
                node=Node.END,
                weight=1000,
                phase=i,
            )

            if (
                self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
                or self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
            ):
                self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, phase=i, key="qdddot", weight=1e-4)

    def _set_constraints(self):
        # --- Constraints --- #
        # Contact force in Z are positive
        self.constraints.add(
            ConstraintFcn.TRACK_CONTACT_FORCES, min_bound=0, max_bound=np.inf, node=Node.ALL, contact_index=1, phase=0
        )  # FP0 > 0 en Z
        if self.nb_phases == 2:
            self.constraints.add(
                ConstraintFcn.TRACK_CONTACT_FORCES,
                min_bound=0,
                max_bound=np.inf,
                node=Node.ALL,
                contact_index=1,
                phase=1,
            )  # FP0 > 0 en Z

        # contact node at zero position and zero speed
        # node = Node.ALL if self.implicit_dynamics else Node.START
        node = Node.START
        self.constraints.add(
            ConstraintFcn.TRACK_MARKERS, node=node, target=self.right_foot_location, marker_index="RFoot", phase=0
        )
        self.constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, node=node, marker_index="RFoot", phase=0)
        if self.nb_phases == 2:
            self.constraints.add(
                ConstraintFcn.TRACK_MARKERS, node=node, target=self.left_foot_location, marker_index="LFoot", phase=1
            )
            self.constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, node=node, marker_index="LFoot", phase=1)

        # first and last step constraints
        self.constraints.add(
            ConstraintFcn.TRACK_MARKERS,
            target=self.initial_left_foot_location,
            node=Node.START,
            marker_index="LFoot",
            phase=0,
        )
        self.constraints.add(
            ConstraintFcn.TRACK_MARKERS,
            target=self.final_left_foot_location,
            node=Node.END,
            marker_index="LFoot",
            phase=0,
        )
        if self.nb_phases == 2:
            # first and last step constraints of the second phase
            self.constraints.add(
                ConstraintFcn.TRACK_MARKERS,
                target=self.initial_right_foot_location,
                node=Node.START,
                marker_index="RFoot",
                phase=1,
            )
            self.constraints.add(
                ConstraintFcn.TRACK_MARKERS,
                target=self.final_right_foot_location,
                node=Node.END,
                marker_index="RFoot",
                phase=1,
            )

        # Ensure lift of foot - Toe Clearance
        if self.has_knee:
            toe_clearance = 0.05
            self.constraints.add(
                ConstraintFcn.TRACK_MARKERS,
                index=2,
                min_bound=toe_clearance - 0.01,
                max_bound=toe_clearance + 0.01,
                target=toe_clearance,
                node=Node.MID,
                marker_index="LFoot",
                phase=0,
            )
            if self.nb_phases == 2:
                self.constraints.add(
                    ConstraintFcn.TRACK_MARKERS,
                    index=2,
                    min_bound=toe_clearance - 0.01,
                    max_bound=toe_clearance + 0.01,
                    target=toe_clearance,
                    node=Node.MID,
                    marker_index="RFoot",
                    phase=1,
                )

    def _set_phase_transition(self):

        idx_q = [0, 1, 2]
        idx_q = idx_q + [3] if self.has_head else idx_q
        idx_qdot = [i + self.biorbd_model[0].nbQ() for i in idx_q]
        idx = idx_q + idx_qdot

        if self.nb_phases == 2:
            idx_cyclic = [i for i in range(self.biorbd_model[0].nbQ() * 2) if i not in idx]
            self.multinode_constraints.add(
                MultinodeConstraintFcn.EQUALITY,
                index=idx_cyclic,
                phase_first_idx=0,
                phase_second_idx=1,
                first_node=Node.END,
                second_node=Node.START,
                weight=1e5,
            )
            self.multinode_constraints.add(
                MultinodeConstraintFcn.EQUALITY,
                index=idx_cyclic,
                phase_first_idx=0,
                phase_second_idx=1,
                first_node=Node.START,
                second_node=Node.END,
                weight=1e5,
            )
            self.multinode_constraints.add(
                MultinodeConstraintFcn.COM_EQUALITY,
                phase_first_idx=0,
                phase_second_idx=1,
                first_node=Node.END,
                second_node=Node.START,
                weight=1e5,
                index=2,
            )
            self.multinode_constraints.add(
                MultinodeConstraintFcn.COM_EQUALITY,
                phase_first_idx=0,
                phase_second_idx=1,
                first_node=Node.START,
                second_node=Node.END,
                weight=1e5,
                index=2,
            )
            self.multinode_constraints.add(
                MultinodeConstraintFcn.COM_VELOCITY_EQUALITY,
                phase_first_idx=0,
                phase_second_idx=1,
                first_node=Node.START,
                second_node=Node.END,
                weight=1e5,
                index=2,
            )
            self.multinode_constraints.add(
                MultinodeConstraintFcn.COM_VELOCITY_EQUALITY,
                phase_first_idx=0,
                phase_second_idx=1,
                first_node=Node.START,
                second_node=Node.END,
                weight=2e5,
                index=2,
            )
            self.phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=0)
        else:
            self.phase_transitions.add(PhaseTransitionFcn.CYCLIC, index=idx, weight=1000)

    def _set_boundary_conditions(self):
        self.x_bounds = BoundsList()
        self.x_bounds.add(
            bounds=QAndQDotAndQDDotBounds(self.biorbd_model[0])
            if self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
            or self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
            else QAndQDotBounds(self.biorbd_model[0])
        )
        if self.nb_phases == 2:
            self.x_bounds.add(
                bounds=QAndQDotAndQDDotBounds(self.biorbd_model[0])
                if self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
                or self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
                else QAndQDotBounds(self.biorbd_model[0])
            )

        nq = self.n_q

        for i in range(self.nb_phases):
            q_sign = 1 if i == 0 else -1
            self.x_bounds[i].max[2, :] = 0  # torso bended forward

            if self.has_head:
                self.x_bounds[i][nq + 3, 0] = 0  # head velocity zero at the beginning
                self.x_bounds[i][nq + 3, -1] = 0  # head velocity zero at the end

            if self.has_knee:
                self.x_bounds[i].min[nq - 2 : nq, [0, -1]] = -np.pi / 8  # driving knees

            # Supervised shoulders
            if self.has_shoulder:
                j = 1 if self.has_head else 0
                self.x_bounds[i][5 + j, 0] = -np.pi / 6 * q_sign
                self.x_bounds[i][6 + j, 0] = np.pi / 6 * q_sign
                self.x_bounds[i][5 + j, -1] = np.pi / 6 * q_sign
                self.x_bounds[i][6 + j, -1] = -np.pi / 6 * q_sign

                self.x_bounds[i][5 + j + nq, 0] = 0
                self.x_bounds[i][5 + j + nq, -1] = 0
                self.x_bounds[i][6 + j + nq, 0] = 0
                self.x_bounds[i][6 + j + nq, -1] = 0

            if self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
                self.u_bounds.add(
                    [self.tau_min] * self.n_tau
                    + [self.qddot_min] * self.n_qddot
                    + [self.fext_min] * self.biorbd_model[0].nbContacts(),
                    [self.tau_max] * self.n_tau
                    + [self.qddot_max] * self.n_qddot
                    + [self.fext_max] * self.biorbd_model[i].nbContacts(),
                )
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS:
                self.u_bounds.add(
                    [self.tau_min] * self.n_tau + [self.qddot_min] * self.n_qddot,
                    [self.tau_max] * self.n_tau + [self.qddot_max] * self.n_qddot,
                )
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK:
                self.u_bounds.add(
                    [self.tau_min] * self.n_tau
                    + [self.qdddot_min] * self.n_qddot
                    + [self.fext_min] * self.biorbd_model[i].nbContacts(),
                    [self.tau_max] * self.n_tau
                    + [self.qdddot_max] * self.n_qddot
                    + [self.fext_max] * self.biorbd_model[i].nbContacts(),
                )
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK:
                self.u_bounds.add(
                    [self.tau_min] * self.n_tau + [self.qdddot_min] * self.n_qddot,
                    [self.tau_max] * self.n_tau + [self.qdddot_max] * self.n_qddot,
                )
            else:
                self.u_bounds.add([self.tau_min] * self.n_tau, [self.tau_max] * self.n_tau)
            # root is not actuated
            self.u_bounds[i][:3, :] = 0

    def _set_initial_guesses(self):
        """
        Set initial guess for the optimization problem.
        """
        # --- Initial guess --- #
        q0 = [0] * self.n_q
        # Torso over the floor and bent
        q0[1] = 0.8
        q0[2] = -np.pi / 6

        for i in range(self.nb_phases):
            if i == 0:
                self.q0i = set_initial_pose(
                    self.biorbd_model_path[0], np.array(q0), self.right_foot_location, self.initial_left_foot_location
                )
                self.q0end = set_initial_pose(
                    self.biorbd_model_path[0], np.array(q0), self.right_foot_location, self.final_left_foot_location
                )
            else:
                self.q0i = set_initial_pose(
                    self.biorbd_model_path[1], np.array(q0), self.initial_right_foot_location, self.left_foot_location
                )
                self.q0end = set_initial_pose(
                    self.biorbd_model_path[1], np.array(q0), self.final_right_foot_location, self.left_foot_location
                )

            # generalized velocities are initialized to 0
            qdot0 = [0] * self.n_qdot

            # concatenate q0 and qdot0
            X0i = []
            X0i.extend(self.q0i)
            X0i.extend(qdot0)
            X0end = []
            X0end.extend(self.q0end)
            X0end.extend(qdot0)
            if (
                self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
                or self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
            ):
                X0i.extend([0] * self.n_qddot)
                X0end.extend([0] * self.n_qddot)
                # X0i.extend([0] * self.n_qddot + [0] * self.biorbd_model[i].nbContacts())
                # X0end.extend([0] * self.n_qddot + [0] * self.biorbd_model[i].nbContacts())

            x = np.linspace(0, self.phase_time[i], 2)
            y = np.array([X0i, X0end]).T
            f = interpolate.interp1d(x, y)
            x_new = np.linspace(0, self.phase_time[i], self.n_shooting[i] + 1)
            X0 = f(x_new)  # use interpolation function returned by `interp1d`

            self._set_initial_states(X0=X0, n_shooting=self.n_shooting[i])
            self._set_initial_controls(n_shooting=self.n_shooting[i])

    def _set_initial_states(self, X0: np.array = None, n_shooting: int = None):
        if X0 is None:
            self.x_init.add([0] * (self.n_q + self.n_q))
        else:
            if X0.shape[1] != n_shooting + 1:
                X0 = self._interpolate_initial_states(X0)

            if not self.ode_solver.is_direct_shooting:
                n = self.ode_solver.polynomial_degree
                X0 = np.repeat(X0, n + 1, axis=1)
                X0 = X0[:, :-n]

            self.x_init.add(X0, interpolation=InterpolationType.EACH_FRAME)

    def _set_initial_controls(self, U0: np.array = None, n_shooting: int = None):
        if U0 is None:
            if self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
                self.u_init.add(
                    [self.tau_init] * self.n_tau
                    + [self.qddot_init] * self.n_qddot
                    + [5] * self.biorbd_model[0].nbContacts()
                )
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK:
                self.u_init.add(
                    [self.tau_init] * self.n_tau
                    + [self.qdddot_init] * self.n_qdddot
                    + [5] * self.biorbd_model[0].nbContacts()
                )
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK:
                self.u_init.add([self.tau_init] * self.n_tau + [self.qdddot_init] * self.n_qdddot)
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS:
                self.u_init.add([self.tau_init] * self.n_tau + [self.qddot_init] * self.n_qddot)
            else:
                self.u_init.add([self.tau_init] * self.n_tau)
        else:
            if U0.shape[1] != n_shooting:
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


def set_initial_pose(model_path: str, q0: np.ndarray, target_RFoot: np.ndarray, target_LFoot: np.ndarray):
    """
    Set the initial pose of the model

    Parameters
    ----------
    model_path : str
        Path to the model
    q0 : np.ndarray
        Initial position of the model
    target_RFoot : np.ndarray
        Target position of the right foot
    target_LFoot : np.ndarray
        Target position of the left foot

    Returns
    -------
    q0 : np.ndarray
        Initial position of the model
    """
    m = biorbd_eigen.Model(model_path)
    bound_min = []
    bound_max = []
    for i in range(m.nbSegment()):
        seg = m.segment(i)
        for r in seg.QRanges():
            bound_min.append(r.min())
            bound_max.append(r.max())
    bounds = (bound_min, bound_max)

    def objective_function(q, *args, **kwargs):
        """
        Objective function to minimize

        Parameters
        ----------
        q : np.ndarray
            Position of the model

        Returns
        -------
        np.ndarray
            Distance between the target position of the right and left foot, and the current position of the right and left foot
        """
        markers = m.markers(q)
        out1 = np.linalg.norm(markers[0].to_array() - target_RFoot) ** 2
        out3 = np.linalg.norm(markers[-1].to_array() - target_LFoot) ** 2

        return out1 + out3

    pos = optimize.least_squares(
        objective_function,
        args=(m, target_RFoot, target_LFoot),
        x0=q0,
        bounds=bounds,
        verbose=1,
        method="trf",
        jac="3-point",
        ftol=1e-10,
        gtol=1e-10,
    )

    return pos.x
