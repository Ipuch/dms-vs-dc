from ezc3d import c3d
import numpy as np
from scipy.interpolate import interp1d
import biorbd


class C3dData:
    """
    The base class for managing c3d file

    Attributes
    ----------
    c3d: ezc3d.c3d
        The c3d file
    marker_names: list
        The list of all marker names in the biorbd model.
    trajectories: ndarray
        The position of the markers

    Methods
    -------
    get_marker_trajectories(self, marker_names: list) -> np.ndarray
        Get markers trajectories
    get_indices(self)
        Get the indices of start and end
    get_final_time(self)
        Get the final time of c3d
    """

    def __init__(self, file_path: str, biorbd_model: biorbd.Model):
        self.c3d = c3d(file_path)
        self.marker_names = [biorbd_model.markerNames()[i].to_string() for i in range(len(biorbd_model.markerNames()))]
        self.trajectories = self.get_marker_trajectories()

    def get_marker_trajectories(self, marker_names: list = None) -> np.ndarray:
        """
        get markers trajectories

        Parameters
        ---------
        marker_names: list
            The list of tracked markers

        Returns
        --------
        an array of markers' position

        """
        marker_names = self.marker_names if marker_names is None else marker_names
        # LOAD C3D FILE
        points = self.c3d["data"]["points"]
        labels_markers = self.c3d["parameters"]["POINT"]["LABELS"]["value"]

        # GET THE MARKERS POSITION (X, Y, Z) AT EACH POINT
        markers = np.zeros((3, len(marker_names), len(points[0, 0, :])))

        for i, name in enumerate(marker_names):
            markers[:, i, :] = points[:3, labels_markers.index(name), :] * 1e-3
        return markers

    def get_indices(self):
        idx_start = 0 + 1
        idx_stop = len(self.trajectories[0, 0, :])
        return [idx_start, idx_stop]

    def get_final_time(self):
        """
        find phase duration
        """
        freq = self.c3d["parameters"]["POINT"]["RATE"]["value"][0]
        index = self.get_indices()
        return [(1 / freq * (index[i + 1] - index[i] + 1)) for i in range(len(index) - 1)][0]


class LoadData:
    """
    The base class for managing c3d data

    Attributes
    ----------
    model: biorbd.Model
        The biorbd Model
    nb_q: int
        The number of generalized coordinates in the model
    nb_qdot: int
        The number of generalized velocities in the model
    nb_markers: int
        The number of markers in the model
    c3d_file: str
        The c3d file path
    c3d_data: C3dData
        The data from c3d
    q: numpy.ndarray
        The array of all values of the q
    qdot: numpy.ndarray
        The array of all values of the qdot

    Methods
    -------
    dispatch_data(self, data, nb_shooting: list, phase_time: list)
        divide and adjust data dimensions to match number of shooting point for each phase
    get_marker_ref(self, number_shooting_points: list, phase_time: list, markers_names: list[str])
        Give the position of the markers from the c3d
    get_experimental_data(self, number_shooting_points, phase_time, with_floating_base: bool)
        Give the values of q, qdot and the position of the markers from the c3d

    """

    def __init__(self, model: biorbd.Model, c3d_file: str, q_file: str, qdot_file: str):
        def load_txt_file(file_path: str):
            data_tp = np.loadtxt(file_path)
            return data_tp

        self.model = model
        self.nb_q = model.nbQ()
        self.nb_qdot = model.nbQdot()
        self.nb_markers = model.nbMarkers()

        # files path
        self.c3d_file = c3d_file
        self.c3d_data = C3dData(c3d_file, model)

        self.nb_frames = self.c3d_data.c3d["parameters"]["POINT"]["FRAMES"]["value"][0]
        self.q = load_txt_file(q_file)
        self.qdot = load_txt_file(qdot_file)
        self.qddot = load_txt_file(qdot_file.removesuffix("qdot.txt") + "qddot.txt")
        self.tau = load_txt_file(qdot_file.removesuffix("qdot.txt") + "tau.txt")

    def dispatch_data(
        self, data: np.ndarray, nb_shooting: list[int], phase_time: list[float], start: int = None, end: int = None
    ) -> list[np.ndarray]:
        """
        divide and adjust data dimensions to match number of shooting point for each phase

        Parameters
        ---------
        data: np.ndarray
            The data we want to adjust
        nb_shooting: list
            The list of nb_shooting for each phase
        phase_time: list
            The list of duration for each phase
        start: int
            The frame number corresponding to the beginning of the studied movement
        end: int
            The frame number corresponding to the end of the studied movement

        Returns
        --------
        out: list[np.ndarray]
            The array of adjusted data

        """

        out = []

        if start and end:
            x = data[:, :, start : end + 1] if len(data.shape) == 3 else data[:, start : end + 1]
            for i in range(end - start):
                t_init = np.linspace(0, (i + 1) / (end + 1 - start), (end + 1 - start))
                t_node = np.linspace(0, (i + 1) / (end + 1 - start), nb_shooting[0] + 1)
                f = interp1d(t_init, x, kind="linear")
                out.append(f(t_node))
        else:
            idx = self.c3d_data.get_indices()
            for i in range(len(nb_shooting)):
                x = data[:, :, idx[i] : idx[i + 1] + 1] if len(data.shape) == 3 else data[:, idx[i] : idx[i + 1] + 1]
                t_init = np.linspace(0, phase_time[i], (idx[i + 1] - idx[i]))
                t_node = np.linspace(0, phase_time[i], nb_shooting[i] + 1)
                f = interp1d(t_init, x, kind="linear")
                out.append(f(t_node))
        return out

    def get_marker_ref(
        self,
        number_shooting_points: list[int],
        phase_time: list[float],
        markers_names: list[str] = None,
        start: int = None,
        end: int = None,
    ):
        """
        divide and adjust the dimensions to match number of shooting point for each phase

        Parameters
        --------
        number_shooting_points: list[int]
            The list of nb_shooting for each phase
        phase_time: list[float]
            The list of duration for each phase
        markers_names: list[str]
            list of tracked markers
        start: int
            The frame number corresponding to the beginning of the studied movement
        end: int
            The frame number corresponding to the end of the studied movement

        Returns
        --------
        The array of marker's position adjusted

        """

        markers = (
            self.c3d_data.trajectories
            if markers_names is None
            else self.c3d_data.get_marker_trajectories(markers_names)
        )

        return self.dispatch_data(
            data=markers,
            nb_shooting=number_shooting_points,
            phase_time=phase_time,
            start=start,
            end=end,
        )

    def get_states_ref(
        self,
        number_shooting_points: list[int],
        phase_time: list[float],
        with_floating_base: bool,
        start: int = None,
        end: int = None,
    ):
        """
        Give all the data from c3d file

        Parameters
        --------
        number_shooting_points: list
            The list of nb_shooting for each phase
        phase_time: list
            The list of duration for each phase
        with_floating_base: bool
            True if there is a floating base in the biorbd model
        start: int
            The frame number corresponding to the beginning of the studied movement
        end: int
            The frame number corresponding to the end of the studied movement

        Returns
        --------
        The values of q and qdot from the c3d

        """
        start = start if start is not None else 0
        end = end if end is not None else self.nb_frames

        q_ref = self.dispatch_data(
            data=self.q,
            nb_shooting=number_shooting_points,
            phase_time=phase_time,
            start=start,
            end=end,
        )
        qdot_ref = self.dispatch_data(
            data=self.qdot,
            nb_shooting=number_shooting_points,
            phase_time=phase_time,
            start=start,
            end=end,
        )
        q_ref[0] = q_ref[0][6:] if not with_floating_base else q_ref[0]
        qdot_ref[0] = qdot_ref[0][6:] if not with_floating_base else qdot_ref[0]
        return q_ref, qdot_ref

    def get_variables_ref(
        self, number_shooting_points: list[int], phase_time: list[float], start: int = None, end: int = None
    ):
        """
        Give all the data from c3d file

        Parameters:
        ---------
        number_shooting_points: list[int]
            The list of nb_shooting for each phase
        phase_time: list[float]
            The list of duration for each phase
        start: int
            The frame number corresponding to the beginning of the studied movement
        end: int
            The frame number corresponding to the end of the studied movement

         Returns
        --------
        The values of q, qdot and tau from the c3d

        """
        start = start if start is not None else 0
        end = end if end is not None else self.nb_frames

        q_ref = self.dispatch_data(
            data=self.q, nb_shooting=number_shooting_points, phase_time=phase_time, start=start, end=end
        )
        qdot_ref = self.dispatch_data(
            data=self.qdot, nb_shooting=number_shooting_points, phase_time=phase_time, start=start, end=end
        )
        tau_ref = self.dispatch_data(
            data=self.tau, nb_shooting=[number_shooting_points[0] - 1], phase_time=phase_time, start=start, end=end
        )
        return q_ref, qdot_ref, tau_ref
