from enum import Enum
from pathlib import Path
import numpy as np

import biorbd_casadi as biorbd
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
    QAndQDotBounds,
    InitialGuess,
)

from ..models.utils import thorax_variables, add_header
from ..models.enums import Models
from ..data.load_events import LoadEvent
from ..data.load_experimental_data import LoadData


class Tasks(Enum):
    """
    Selection of tasks
    """

    ARMPIT = Path(__file__).parent.parent.__str__() + "data/F0_aisselle_05"
    DRINK = Path(__file__).parent.parent.__str__() + "data/F0_boire_05"
    TEETH = Path(__file__).parent.parent.__str__() + "data/F0_dents_05"
    DRAW = Path(__file__).parent.parent.__str__() + "data/F0_dessiner_05"
    EAT = Path(__file__).parent.parent.__str__() + "data/F0_manger_05"
    HEAD = Path(__file__).parent.parent.__str__() + "data/F0_tete_05"


class UpperLimbOCP:
    """
    Class to generate the OCP for the upper limb motion for a daily task of living

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
        n_shooting: int = 125,
        phase_durations: float = 1.50187,  # actualized with results from https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4096894
        n_threads: int = 8,
        ode_solver: OdeSolver = OdeSolver.RK4(),
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        task: Tasks = Tasks.TEETH,
        use_sx: bool = False,
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
        self.n_shooting = n_shooting

        self.task = task

        self.x = None
        self.u = None

        self.phase_durations = phase_durations
        self.phase_time = phase_durations

        self.duration = np.sum(self.phase_durations)

        self.n_threads = n_threads
        self.ode_solver = ode_solver

        if biorbd_model_path is not None:

            self.biorbd_model = (biorbd.Model(biorbd_model_path),)
            self.marker_labels = [m.to_string() for m in self.biorbd_model.markerNames()]
            self.rigidbody_dynamics = rigidbody_dynamics

            self.n_q = self.biorbd_model[0].nbQ()
            self.n_qdot = self.biorbd_model[0].nbQdot()
            self.nb_root = self.biorbd_model[0].nbRoot()

            self.n_tau = (
                self.biorbd_model[0].nbGeneralizedTorque()
                - self.biorbd_model[0].nbRoot()
            )

            self.tau_min, self.tau_init, self.tau_max = -50, 0, 50
            self.muscle_min, self.muscle_max, self.muscle_init = 0, 1, 0.05

            self.velocity_max = 100  # qdot
            self.velocity_max_phase_transition = 10  # qdot hips, thorax in phase 2

            self.random_scale = (
                0.02  # relative to the maximal bounds of the states or controls
            )
            self.random_scale_qdot = 0.02
            self.random_scale_qddot = 0.02
            self.random_scale_tau = 0.02

            self.dynamics = DynamicsList()
            self.constraints = ConstraintList()
            self.objective_functions = ObjectiveList()

            self.x_bounds = BoundsList()
            self.u_bounds = BoundsList()

            self.initial_states = []

            self.x_init = InitialGuessList() if initial_x is None else initial_x
            self.u_init = InitialGuessList() if initial_u is None else initial_u

            self._set_boundary_conditions()

            np.random.seed(seed)
            # todo

            if initial_x is None:
                self._set_initial_guesses()  # noise is into the initial guess
            if initial_u is None:
                self._set_initial_controls()  # noise is into the initial guess

            self._set_dynamics()
            self._set_objective_functions()

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
                n_threads=n_threads,
                control_type=ControlType.CONSTANT,
                ode_solver=ode_solver,
                use_sx=use_sx,
            )

    def _get_experimental_data(self):
        n_shooting_points = 100

        data_path = c3d_path.removesuffix(c3d_path.split("/")[-1])
        file_path = data_path + Models.WU_INVERSE_KINEMATICS_XYZ.name + "_" + c3d_path.split("/")[-1].removesuffix(
            ".c3d")
        q_file_path = file_path + "_q.txt"
        qdot_file_path = file_path + "_qdot.txt"

        thorax_values = thorax_variables(q_file_path)  # load c3d floating base pose
        model_template_path = Models.WU_WITHOUT_FLOATING_BASE_QUAT_DEGROOTE_TEMPLATE.value
        new_biomod_file = Models.WU_WITHOUT_FLOATING_BASE_QUAT_DEGROOTE_VARIABLES.value
        add_header(model_template_path, new_biomod_file, thorax_values)

        biorbd_model = biorbd.Model(new_biomod_file)
        marker_ref = [m.to_string() for m in biorbd_model.markerNames()]

        # get key events
        event = LoadEvent(c3d_path=c3d_path, marker_list=marker_ref)
        data = LoadData(biorbd_model, c3d_path, q_file_path, qdot_file_path)
        if c3d_path == Tasks.EAT.value:
            start_frame = event.get_frame(1)
            end_frame = event.get_frame(2)
            phase_time = event.get_time(2) - event.get_time(1)
        else:
            start_frame = event.get_frame(0)
            end_frame = event.get_frame(1)
            phase_time = event.get_time(1) - event.get_time(0)
        target = data.get_marker_ref(
            number_shooting_points=[n_shooting_points],
            phase_time=[phase_time],
            start=int(start_frame),
            end=int(end_frame),
        )

        # load initial guesses
        q_ref, qdot_ref, tau_ref = data.get_variables_ref(
            number_shooting_points=[n_shooting_points],
            phase_time=[phase_time],
            start=int(start_frame),
            end=int(end_frame),
        )
        x_init_ref = np.concatenate([q_ref[0][6:, :], qdot_ref[0][6:, :]])  # without floating base
        u_init_ref = tau_ref[0][6:, :]
        nb_q = biorbd_model.nbQ()
        nb_qdot = biorbd_model.nbQdot()
        x_init_quat = np.vstack((np.zeros((nb_q, n_shooting_points + 1)), np.ones((nb_qdot, n_shooting_points + 1))))
        for i in range(n_shooting_points + 1):
            x_quat_shoulder = eul2quat(x_init_ref[5:8, i])
            x_init_quat[5:8, i] = x_quat_shoulder[1:]
            x_init_quat[10, i] = x_quat_shoulder[0]
        x_init_quat[:5] = x_init_ref[:5]
        x_init_quat[8:10] = x_init_ref[8:10]
        x_init_quat[11:, :] = x_init_ref[10:, :]

    def _set_dynamics(self):
        """
        Set the dynamics of the optimal control problem
        """

        if self.rigidbody_dynamics == RigidBodyDynamics.ODE:
            dynamics = Dynamics(DynamicsFcn.MUSCLE_DRIVEN, with_torque=True)
        else:
            raise ValueError("This dynamics has not been implemented")

    def _set_objective_functions(self):
        """
        Set the multi-objective functions for each phase with specific weights
        """

        if self.task == Tasks.TEETH:
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=range(5), weight=200)  # added
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=5)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", derivative=True, weight=0.5)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1000)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=15)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1500)

        elif self.task == Tasks.EAT or self.task == Tasks.HEAD:
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=range(5), weight=200)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=50)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", derivative=True,
                                    weight=0.5)  # added
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1000)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=15)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1500)

        elif self.task == Tasks.ARMPIT:
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=range(5), weight=200)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=800)  # was 5
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", derivative=True,
                                    weight=0.5)  # added
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1000)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1500)

        elif self.task == Tasks.DRINK:
            # converges but solution isn't adequate yet
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=range(5), weight=200)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=150)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", derivative=True, weight=0.5)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1000)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1500)
            # tried minimizing the derivative of qdot without minimizing qdot
            # tried minimizing tau index = 8

        else:
            raise NotImplementedError("This task is not implemented yet.")

    def _set_initial_guesses(self):
        """
        Set the initial guess for the optimal control problem (states and controls)
        """
        # --- Initial guess --- #
        # todo
        self.x_init = InitialGuess(x_init_ref, interpolation=InterpolationType.EACH_FRAME)
        self.u_init = InitialGuess([tau_init] * n_torque + [muscle_init] * biorbd_model.nbMuscles())


    def _set_initial_states(self, X0: np.array = None):
        """
        Set the initial states of the optimal control problem.
        """
        X0 = (
            np.zeros((self.n_q + self.n_qdot, self.n_shooting + 1))
            if X0 is None
            else X0
        )
        self.x_init.add(X0, interpolation=InterpolationType.EACH_FRAME)

    def _set_initial_controls(self, U0: np.array = None):
        if U0 is None and self.u is None:
            n_shooting = self.n_shooting
            tau_J_random = np.random.random((self.n_tau, n_shooting)) * 2 - 1

            tau_max = self.tau_max * np.ones(self.n_tau)
            tau_max[self.high_torque_idx] = self.tau_hips_max
            tau_J_random = tau_J_random * tau_max[:, np.newaxis] * self.random_scale_tau

            qddot_J_random = (
                (np.random.random((self.n_tau, n_shooting)) * 2 - 1)
                * self.qddot_max
                * self.random_scale_qddot
            )
            qddot_B_random = (
                (np.random.random((self.nb_root, n_shooting)) * 2 - 1)
                * self.qddot_max
                * self.random_scale_qddot
            )

            if self.rigidbody_dynamics == RigidBodyDynamics.ODE:
                self.u_init.add(
                    tau_J_random, interpolation=InterpolationType.EACH_FRAME
                )
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
        self.x_bounds = QAndQDotBounds(self.biorbd_model)
        self.x_bounds.min[:self.n_q, 0] = x_init_ref[:self.n_q, 0] - 0.1 * np.ones(x_init_ref[:self.n_q, 0].shape)
        self.x_bounds.max[:self.n_q, 0] = x_init_ref[:self.n_q, 0] + 0.1 * np.ones(x_init_ref[:self.n_q, 0].shape)
        self.x_bounds.min[:self.n_q, -1] = x_init_ref[:self.n_q, -1] - 0.1 * np.ones(x_init_ref[:self.n_q, -1].shape)
        self.x_bounds.max[:self.n_q, -1] = x_init_ref[:self.n_q, -1] + 0.1 * np.ones(x_init_ref[:self.n_q, -1].shape)

        # norm of the quaternion should be 1 at the start and at the end
        self.x_bounds.min[5:8, 0] = x_init_ref[5:8, 0]
        self.x_bounds.max[5:8, 0] = x_init_ref[5:8, 0]
        self.x_bounds.min[5:8, -1] = x_init_ref[5:8, -1]
        self.x_bounds.max[5:8, -1] = x_init_ref[5:8, -1]

        self.x_bounds.min[10, 0] = x_init_ref[10, 0]
        self.x_bounds.max[10, 0] = x_init_ref[10, 0]
        self.x_bounds.min[10, -1] = x_init_ref[10, -1]
        self.x_bounds.max[10, -1] = x_init_ref[10, -1]

        self.x_bounds.min[self.n_q:, 0] = [-1e-3] * self.biorbd_model.nbQdot()
        self.x_bounds.max[self.n_q:, 0] = [1e-3] * self.biorbd_model.nbQdot()
        self.x_bounds.min[self.n_q:, -1] = [-1e-1] * self.biorbd_model.nbQdot()
        self.x_bounds.max[self.n_q:, -1] = [1e-1] * self.biorbd_model.nbQdot()
        self.x_bounds.min[8:10, 1], self.x_bounds.min[10, 1] = self.x_bounds.min[9:11, 1], -1
        self.x_bounds.max[8:10, 1], self.x_bounds.max[10, 1] = self.x_bounds.max[9:11, 1], 1

        if self.rigidbody_dynamics == RigidBodyDynamics.ODE:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau + [self.muscle_min] * self.biorbd_model.nbMuscleTotal(),
                [self.tau_max] * self.n_tau + [self.muscle_max] * self.biorbd_model.nbMuscleTotal(),
            )
            self.u_bounds[0][5:8] = 0
            self.u_bounds[0][5:8] = 0



