import ezc3d
from pyomeca import Markers
import numpy as np


class LoadEvent:
    def __init__(self,
                 c3d_path: str,
                 marker_list: list[str],
                 ):
        self.c3d_path = c3d_path
        self.c3d = ezc3d.c3d(c3d_path)
        self.marker_list = marker_list

    def get_time(self, idx: int) -> np.ndarray:
        """
        find the time corresponding to the event

        Parameters
        ---------
        idx: int
            index number of the event

        Returns
        --------
        event_values: ndarray
            array with the time value in seconds

        """
        event_time = self.c3d["parameters"]["EVENT"]["TIMES"]["value"][1][idx]
        return np.array(event_time)

    def get_frame(self, idx: int) -> np.ndarray:
        """
        find the frame corresponding to the event

        Parameters
        ---------
        idx: int
            index number of the event

        Returns
        --------
        event_values: ndarray
            array with the frame number

        """
        frame_rate = self.c3d["parameters"]["TRIAL"]["CAMERA_RATE"]["value"][0]
        frame = round(self.get_time(idx) * frame_rate)
        start_frame = self.c3d["parameters"]["TRIAL"]["ACTUAL_START_FIELD"]["value"][0]
        event_frame = frame - start_frame
        return np.array(event_frame)

    def get_markers(self, idx: int) -> np.ndarray:
        """
        find the position of each marker during an event

        Parameters
        ---------
        idx: int
            index number of the event

        Returns
        --------
        event_values: ndarray
            array with the position along three axes of each marker in millimeters

        """

        markers = Markers.from_c3d(self.c3d_path, usecols=self.marker_list, prefix_delimiter=":").to_numpy()
        event_markers = markers[:3, :, self.get_frame(idx)]

        return event_markers

    def get_event(self, idx: int) -> dict:
        """
        find the time, the frame and the position of each marker during an event

        Parameters
        ---------
        idx: int
            index number of the event

        Returns
        --------
        event_values: dict
            dictionary containing the time, the frame and the positions of each marker for the event corresponding to
            the given index

        """

        event_values = {"time": self.get_time(idx), "frame": self.get_frame(idx), "markers": self.get_markers(idx)}

        return event_values




