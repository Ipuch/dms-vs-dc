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
    Dynamics,
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


def eul2quat(eul: np.ndarray) -> np.ndarray:
    rotation_matrix = biorbd.Rotation_fromEulerAngles(eul, "xyz")
    quat = biorbd.Quaternion_fromMatrix(rotation_matrix).to_array().squeeze()
    return quat


def quat2eul(quat: np.ndarray) -> np.ndarray:
    quat_biorbd = biorbd.Quaternion(quat[0], quat[1], quat[2], quat[3])
    rotation_matrix = biorbd.Quaternion.toMatrix(quat_biorbd)
    eul = biorbd.Rotation_toEulerAngles(rotation_matrix, "xyz").to_array()
    return eul


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
        n_shooting: int = 100,
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
        self.c3d_path = f"{self.task.value}.c3d"

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
                self._get_experimental_data()
                self._set_initial_guesses()  # noise is into the initial guess
            self._set_boundary_conditions()

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

        data_path = self.task
        file_path = data_path + Models.WU_INVERSE_KINEMATICS_XYZ.name + "_" + self.task
        q_file_path = file_path + "_q.txt"
        qdot_file_path = file_path + "_qdot.txt"

        thorax_values = thorax_variables(q_file_path)  # load c3d floating base pose
        model_template_path = Models.WU_WITHOUT_FLOATING_BASE_QUAT_DEGROOTE_TEMPLATE.value
        new_biomod_file = Models.WU_WITHOUT_FLOATING_BASE_QUAT_DEGROOTE_VARIABLES.value
        add_header(model_template_path, new_biomod_file, thorax_values)

        biorbd_model = biorbd.Model(new_biomod_file)
        marker_ref = [m.to_string() for m in biorbd_model.markerNames()]

        # get key events
        event = LoadEvent(c3d_path=self.c3d_path, marker_list=marker_ref)
        if self.c3d_path == Tasks.EAT.value:
            start_frame = event.get_frame(1)
            end_frame = event.get_frame(2)
            phase_time = event.get_time(2) - event.get_time(1)
        else:
            start_frame = event.get_frame(0)
            end_frame = event.get_frame(1)
            phase_time = event.get_time(1) - event.get_time(0)

        # get target
        data = LoadData(biorbd_model, self.c3d_path, q_file_path, qdot_file_path)
        target = data.get_marker_ref(
            number_shooting_points=[self.n_shooting],
            phase_time=[phase_time],
            start=int(start_frame),
            end=int(end_frame),
        )

        # load initial guesses
        q_ref, qdot_ref, tau_ref = data.get_variables_ref(
            number_shooting_points=[self.n_shooting],
            phase_time=[phase_time],
            start=int(start_frame),
            end=int(end_frame),
        )

        # building initial guess
        x_init_ref = np.concatenate([q_ref[0][6:, :], qdot_ref[0][6:, :]])  # without floating base

        self.u_init_ref = tau_ref[0][6:, :]

        nb_q = biorbd_model.nbQ()
        nb_qdot = biorbd_model.nbQdot()
        x_init_quat = np.vstack((np.zeros((nb_q, self.n_shooting + 1)), np.ones((nb_qdot, self.n_shooting + 1))))
        for i in range(self.n_shooting + 1):
            x_quat_shoulder = eul2quat(x_init_ref[5:8, i])
            x_init_quat[5:8, i] = x_quat_shoulder[1:]
            x_init_quat[10, i] = x_quat_shoulder[0]
        x_init_quat[:5] = x_init_ref[:5]
        x_init_quat[8:10] = x_init_ref[8:10]
        x_init_quat[11:, :] = x_init_ref[10:, :]

        self.x_init_ref = x_init_quat

    def _set_dynamics(self):
        """
        Set the dynamics of the optimal control problem
        """

        if self.rigidbody_dynamics == RigidBodyDynamics.ODE:
            self.dynamics = Dynamics(DynamicsFcn.MUSCLE_DRIVEN, with_torque=True)
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
        self.x_init = InitialGuess(self.x_init_ref, interpolation=InterpolationType.EACH_FRAME)

        self.u_init = InitialGuess([self.tau_init] * self.n_tau + [self.muscle_init] * self.biorbd_model.nbMuscles())
        self.u_init = InitialGuess(self.u_init_ref, interpolation=InterpolationType.EACH_FRAME)

    def _set_boundary_conditions(self):
        """
        Set the boundary conditions for controls and states for each phase.
        """
        self.x_bounds = QAndQDotBounds(self.biorbd_model)
        x_slack_start = 0.1 * np.ones(self.x_init_ref[:self.n_q, 0].shape)
        self.x_bounds.min[:self.n_q, 0] = self.x_init_ref[:self.n_q, 0] - x_slack_start
        self.x_bounds.max[:self.n_q, 0] = self.x_init_ref[:self.n_q, 0] + x_slack_start

        x_slack_end = 0.1 * np.ones(self.x_init_ref[:self.n_q, -1].shape)
        self.x_bounds.min[:self.n_q, -1] = self.x_init_ref[:self.n_q, -1] - x_slack_end
        self.x_bounds.max[:self.n_q, -1] = self.x_init_ref[:self.n_q, -1] + x_slack_end

        # norm of the quaternion should be 1 at the start and at the end
        self.x_bounds.min[5:8, 0] = self.x_init_ref[5:8, 0]
        self.x_bounds.max[5:8, 0] = self.x_init_ref[5:8, 0]
        self.x_bounds.min[5:8, -1] = self.x_init_ref[5:8, -1]
        self.x_bounds.max[5:8, -1] = self.x_init_ref[5:8, -1]

        self.x_bounds.min[10, 0] = self.x_init_ref[10, 0]
        self.x_bounds.max[10, 0] = self.x_init_ref[10, 0]
        self.x_bounds.min[10, -1] = self.x_init_ref[10, -1]
        self.x_bounds.max[10, -1] = self.x_init_ref[10, -1]

        self.x_bounds.min[self.n_q:, 0] = [-1e-3] * self.biorbd_model.nbQdot()
        self.x_bounds.max[self.n_q:, 0] = [1e-3] * self.biorbd_model.nbQdot()
        self.x_bounds.min[self.n_q:, -1] = [-1e-1] * self.biorbd_model.nbQdot()
        self.x_bounds.max[self.n_q:, -1] = [1e-1] * self.biorbd_model.nbQdot()
        self.x_bounds.min[8:10, 1], self.x_bounds.min[10, 1] = self.x_bounds.min[9:11, 1], -1
        self.x_bounds.max[8:10, 1], self.x_bounds.max[10, 1] = self.x_bounds.max[9:11, 1], 1

        self.u_bounds.add(
            [self.tau_min] * self.n_tau + [self.muscle_min] * self.biorbd_model.nbMuscleTotal(),
            [self.tau_max] * self.n_tau + [self.muscle_max] * self.biorbd_model.nbMuscleTotal(),
        )
        self.u_bounds[0][5:8] = 0
        self.u_bounds[0][5:8] = 0



