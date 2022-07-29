from typing import Callable, Any, Union
from copy import deepcopy, copy

import numpy as np
from scipy.integrate import solve_ivp
from scipy import interpolate as sci_interp

from bioptim import SolutionIntegrator, Shooting, OptimalControlProgram, Solution, ControlType
import biorbd


class Integration:
    """
    Integration

    Attributes
    ----------
    ocp: SimplifiedOCP
        The OCP simplified
    ns: list
        The number of shooting point for each phase
    is_interpolated: bool
        If the current structure is interpolated
    is_integrated: bool
        If the current structure is integrated
    is_merged: bool
        If the phases were merged
    vector: np.ndarray
        The data in the vector format
    _states: list
        The data structure that holds the states
    _controls: list
        The data structure that holds the controls
    parameters: dict
        The data structure that holds the parameters
    phase_time: list
        The total time for each phases

    Methods
    -------
    copy(self, skip_data: bool = False) -> Any
        Create a deepcopy of the Solution
    @property
    states(self) -> Union[list, dict]
        Returns the state in list if more than one phases, otherwise it returns the only dict
    @property
    controls(self) -> Union[list, dict]
        Returns the controls in list if more than one phases, otherwise it returns the only dict
    integrate(self, shooting_type: Shooting = Shooting.MULTIPLE, keep_intermediate_points: bool = True,
              merge_phases: bool = False, continuous: bool = True) -> Solution
        Integrate the states
    interpolate(self, n_frames: Union[int, list, tuple]) -> Solution
        Interpolate the states
    merge_phases(self) -> Solution
        Get a data structure where all the phases are merged into one
    _merge_phases(self, skip_states: bool = False, skip_controls: bool = False) -> tuple
        Actually performing the phase merging
    _complete_control(self)
        Controls don't necessarily have dimensions that matches the states. This method aligns them
    graphs(self, automatically_organize: bool, show_bounds: bool,
           show_now: bool, shooting_type: Shooting)
        Show the graphs of the simulation
    animate(self, n_frames: int = 0, show_now: bool = True, **kwargs: Any) -> Union[None, list]
        Animate the simulation
    print(self, cost_type: CostType = CostType.ALL)
        Print the objective functions and/or constraints to the console
    """

    def __init__(
        self,
        ocp: OptimalControlProgram,
        solution: Solution,
        state_keys: list = None,
        control_keys: list = None,
        fext_keys: list = None,
        function: Callable = None,
        **extra_variables,
    ):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            The OCP
        state_keys: list
            The state keys
        control_keys: list
            The control keys
        fext_keys: list
            The external forces keys
        function: Callable
            The function that will be used to evaluate the system dynamics
        extra_variables: dict
            The extra variables that will be used to evaluate the system dynamics
        """

        self.is_integrated = False
        self.is_interpolated = False
        self.is_merged = False
        self.recomputed_time_steps = False

        self.ocp = ocp
        self.ns = [nlp.ns for nlp in self.ocp.nlp]
        self.model = [biorbd.Model(nlp.model.path().absolutePath().to_string()) for nlp in self.ocp.nlp]

        self.control_keys = control_keys
        self.state_keys = state_keys
        self.fext_keys = fext_keys

        # Extract the data now for further use
        self._states = self._update_variable_with_keys(solution._states, self.state_keys)
        self._controls = self._update_variable_with_keys(solution._controls, self.control_keys)
        self._fext = (
            self._update_variable_with_keys(solution._controls, self.fext_keys) if self.fext_keys is not None else None
        )
        self.parameters = solution.parameters
        self.vector = solution.vector
        self.time_vector = None

        self.phase_time = solution.phase_time
        self.function = function if function is not None else [nlp.dynamics for nlp in self.ocp.nlp]

        self.mode = extra_variables["mode"] if "mode" in extra_variables else None

    @staticmethod
    def _update_variable_with_keys(variable: list[dict], keys: list) -> list:
        """
        Update the variable removing the key of the dictionary
        """
        cleaned_list = [{k: v for k, v in d.items() if k in keys} for d in variable]
        # add all key gathering all the keys in one item
        for i, phase_list in enumerate(cleaned_list):
            # empty numpy array named all_variables
            all_variables = np.empty((0, 0))
            # concatenate all the data of all items of the dictionary
            for key, value in phase_list.items():
                all_variables = value if all_variables.size == 0 else np.concatenate((all_variables, value), axis=0)

            # add the data to the dictionary
            cleaned_list[i]["all"] = all_variables

        return cleaned_list

    @property
    def states(self) -> Union[list, dict]:
        """
        Returns the state in list if more than one phases, otherwise it returns the only dict

        Returns
        -------
        The states data
        """

        return self._states[0] if len(self._states) == 1 else self._states

    @property
    def controls(self) -> Union[list, dict]:
        """
        Returns the controls in list if more than one phases, otherwise it returns the only dict

        Returns
        -------
        The controls data
        """

        if not self._controls:
            raise RuntimeError(
                "There is no controls in the solution. "
                "This may happen in "
                "previously integrated and interpolated structure"
            )
        return self._controls[0] if len(self._controls) == 1 else self._controls

    def integrate(
        self,
        shooting_type: Shooting = Shooting.SINGLE_CONTINUOUS,
        keep_intermediate_points: bool = False,
        merge_phases: bool = False,
        continuous: bool = True,
        integrator: SolutionIntegrator = SolutionIntegrator.DEFAULT,
    ) -> Any:
        """
        Integrate the states

        Parameters
        ----------
        shooting_type: Shooting
            Which type of integration
        keep_intermediate_points: bool
            If the integration should returns the intermediate values of the integration [False]
            or only keep the node [True] effective keeping the initial size of the states
        merge_phases: bool
            If the phase should be merged in a unique phase
        continuous: bool
            If the arrival value of a node should be discarded [True] or kept [False]. The value of an integrated
            arrival node and the beginning of the next one are expected to be almost equal when the problem converged
        integrator: SolutionIntegrator
            Use the ode defined by OCP or use a separate integrator provided by scipy

        Returns
        -------
        A Solution data structure with the states integrated. The controls are removed from this structure
        """

        # Sanity check
        if self.is_integrated:
            raise RuntimeError("Cannot integrate twice")
        if self.is_interpolated:
            raise RuntimeError("Cannot integrate after interpolating")
        if self.is_merged:
            raise RuntimeError("Cannot integrate after merging phases")

        if shooting_type == Shooting.MULTIPLE and not keep_intermediate_points:
            raise ValueError(
                "Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously "
                "since it would do nothing"
            )
        if shooting_type == Shooting.SINGLE_CONTINUOUS and not continuous:
            raise ValueError(
                "Shooting.SINGLE_CONTINUOUS and continuous=False cannot be used simultaneously it is a contradiction"
            )

        out = self.__perform_integration(shooting_type, keep_intermediate_points, continuous, merge_phases, integrator)

        if merge_phases:
            if continuous:
                out = out.interpolate(sum(out.ns) + 1)
            else:
                out._states, _, out.phase_time, out.ns = out._merge_phases(skip_controls=True, continuous=continuous)
                out.is_merged = True
        out.is_integrated = True

        return out

    def _generate_time_vector(
        self,
        time_phase,
        keep_intermediate_points: bool,
        continuous: bool,
        merge_phases: bool,
        integrator: SolutionIntegrator,
    ):
        """
        Generate time integration vector, at which the points from intagrate are evaluated

        """

        t_integrated = []
        last_t = 0
        for phase_idx, nlp in enumerate(self.ocp.nlp):
            n_int_steps = (
                nlp.ode_solver.steps_scipy if integrator != SolutionIntegrator.DEFAULT else nlp.ode_solver.steps
            )
            dt_ns = time_phase[phase_idx + 1] / nlp.ns
            time_phase_integrated = []
            last_t_int = copy(last_t)
            for _ in range(nlp.ns):
                if nlp.ode_solver.is_direct_collocation and integrator == SolutionIntegrator.DEFAULT:
                    time_phase_integrated += (np.array(nlp.dynamics[0].step_time) * dt_ns + last_t_int).tolist()
                else:
                    time_interval = np.linspace(last_t_int, last_t_int + dt_ns, n_int_steps + 1)
                    if continuous and _ != nlp.ns - 1:
                        time_interval = time_interval[:-1]
                    if not keep_intermediate_points:
                        if _ == nlp.ns - 1:
                            time_interval = time_interval[[0, -1]]
                        else:
                            time_interval = np.array([time_interval[0]])
                    time_phase_integrated += time_interval.tolist()

                if not continuous and _ == nlp.ns - 1:
                    time_phase_integrated += [time_phase_integrated[-1]]

                last_t_int += dt_ns
            if continuous and merge_phases and phase_idx != len(self.ocp.nlp) - 1:
                t_integrated += time_phase_integrated[:-1]
            else:
                t_integrated += time_phase_integrated
            last_t += time_phase[phase_idx + 1]
        return t_integrated

    def copy(self, skip_data: bool = False) -> Any:
        """
        Create a deepcopy of the Solution

        Parameters
        ----------
        skip_data: bool
            If data should be ignored in the copy

        Returns
        -------
        Return a Solution data structure
        """

        new = Solution(self.ocp, None)

        new.vector = deepcopy(self.vector)

        new.is_interpolated = deepcopy(self.is_interpolated)
        new.is_integrated = deepcopy(self.is_integrated)
        new.is_merged = deepcopy(self.is_merged)

        new.phase_time = deepcopy(self.phase_time)
        new.ns = deepcopy(self.ns)

        new.time_vector = deepcopy(self.time_vector)

        if skip_data:
            new._states, new._controls, new.parameters = [], [], {}
        else:
            new._states = deepcopy(self._states)
            new._controls = deepcopy(self._controls)
            new.parameters = deepcopy(self.parameters)

        return new

    def __perform_integration(
        self,
        shooting_type: Shooting,
        keep_intermediate_points: bool,
        continuous: bool,
        merge_phases: bool,
        integrator: SolutionIntegrator,
    ):
        n_direct_collocation = sum([nlp.ode_solver.is_direct_collocation for nlp in self.ocp.nlp])

        if n_direct_collocation > 0 and integrator == SolutionIntegrator.DEFAULT:
            if continuous:
                raise RuntimeError(
                    "Integration with direct collocation must be not continuous if a scipy integrator is used"
                )

            if shooting_type != Shooting.MULTIPLE:
                raise RuntimeError(
                    "Integration with direct collocation must using shooting_type=Shooting.MULTIPLE "
                    "if a scipy integrator is not used"
                )

        # Copy the data
        out = self.copy(skip_data=True)
        out.recomputed_time_steps = integrator != SolutionIntegrator.DEFAULT
        out._states = []
        out.time_vector = self._generate_time_vector(
            out.phase_time, keep_intermediate_points, continuous, merge_phases, integrator
        )
        for _ in range(len(self._states)):
            out._states.append({})

        sum_states_len = 0
        params = self.parameters["all"]
        x0 = self._states[0]["all"][:, 0]

        def get_u(u, t, t_interval):
            return u[:, 0] + (u[:, 1] - u[:, 0]) * t / (t_interval[1] - t_interval[0])

        for p, nlp in enumerate(self.ocp.nlp):
            param_scaling = nlp.parameters.scaling
            n_states = self._states[p]["all"].shape[0]
            n_steps = nlp.ode_solver.steps_scipy if integrator != SolutionIntegrator.DEFAULT else nlp.ode_solver.steps
            if not continuous:
                n_steps += 1
            if keep_intermediate_points:
                out.ns[p] *= n_steps

            out._states[p]["all"] = np.ndarray((n_states, out.ns[p] + 1))

            # Get the first frame of the phase
            if shooting_type == Shooting.SINGLE_CONTINUOUS:
                if p != 0:
                    u0 = self._controls[p - 1]["all"][:, -1]
                    val = self.ocp.phase_transitions[p - 1].function(np.vstack((x0, x0)), np.vstack((u0, u0)), params)
                    if val.shape[0] != x0.shape[0]:
                        raise RuntimeError(
                            f"Phase transition must have the same number of states ({val.shape[0]}) "
                            f"when integrating with Shooting.SINGLE_CONTINUOUS. If it is not possible, "
                            f"please integrate with Shooting.SINGLE"
                        )
                    x0 += np.array(val)[:, 0]
            else:
                col = (
                    slice(0, n_steps)
                    if nlp.ode_solver.is_direct_collocation and integrator == SolutionIntegrator.DEFAULT
                    else 0
                )
                x0 = self._states[p]["all"][:, col]

            for s in range(self.ns[p]):
                # print(s)
                # if s == self.ns[p] - 1:
                # print(self._controls[p]["all"][:, s: s + 2])
                # print("stop")

                if self.mode is not None:
                    if self.mode == "constant_control":
                        u = self._controls[p]["all"][:, s]
                    elif self.mode == "linear_control":
                        u = self._controls[p]["all"][:, s : s + 2]
                        if np.isnan(u[:, 1]).all():
                            u[:, 1] = 0
                    else:
                        raise NotImplementedError(f"mode {self.mode} " f"not yet implemented in integrating")

                else:
                    if nlp.control_type == ControlType.CONSTANT:
                        u = self._controls[p]["all"][:, s]
                    elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                        u = self._controls[p]["all"][:, s : s + 2]
                        # check if the last colmuns is full of nans
                    else:
                        raise NotImplementedError(
                            f"ControlType {nlp.control_type} " f"not yet implemented in integrating"
                        )

                fext = (
                    self._fext[p]["all"][:, s]
                    if self._fext is not None and self._fext[p]["all"].shape != (0, 0)
                    else None
                )

                if integrator != SolutionIntegrator.DEFAULT:
                    t_init = sum(out.phase_time[:p]) / nlp.ns
                    t_end = sum(out.phase_time[: (p + 2)]) / nlp.ns
                    n_points = n_steps + 1 if continuous else n_steps
                    t_eval = np.linspace(t_init, t_end, n_points) if keep_intermediate_points else [t_init, t_end]

                    if self.mode == "constant_control":
                        f_lambda = lambda t, x: np.array(self.function(self.model[p], x, u, params, fext))
                    else:
                        f_lambda = lambda t, x: np.array(
                            self.function(self.model[p], x, get_u(u, t, [t_init, t_end]), params, fext)
                        )

                    integrated = solve_ivp(
                        # lambda t, x: np.array(nlp.dynamics_func(x, u, params))[:, 0],
                        f_lambda,
                        [t_init, t_end],
                        x0,
                        t_eval=t_eval,
                        method=integrator.value,
                    ).y

                    next_state_col = (
                        (s + 1) * (nlp.ode_solver.steps + 1) if nlp.ode_solver.is_direct_collocation else s + 1
                    )
                    cols_in_out = [s * n_steps, (s + 1) * n_steps] if keep_intermediate_points else [s, s + 2]
                    # else:
                    # if nlp.ode_solver.is_direct_collocation:
                    #     if keep_intermediate_points:
                    #         integrated = x0  # That is only for continuous=False
                    #         cols_in_out = [s * n_steps, (s + 1) * n_steps]
                    #     else:
                    #         integrated = x0[:, [0, -1]]
                    #         cols_in_out = [s, s + 2]
                    #     next_state_col = slice((s + 1) * n_steps, (s + 2) * n_steps)
                    #
                    # else:
                    #     if keep_intermediate_points:
                    #         integrated = np.array(nlp.dynamics[s](x0=x0, p=u, params=params / param_scaling)["xall"])
                    #         cols_in_out = [s * n_steps, (s + 1) * n_steps]
                    #     else:
                    #         integrated = np.concatenate(
                    #             (x0[:, np.newaxis], nlp.dynamics[s](x0=x0, p=u, params=params / param_scaling)["xf"]),
                    #             axis=1,
                    #         )
                    #         cols_in_out = [s, s + 2]
                    #     next_state_col = s + 1

                    cols_in_out = slice(
                        cols_in_out[0],
                        cols_in_out[1] + 1 if continuous and keep_intermediate_points else cols_in_out[1],
                    )
                    out._states[p]["all"][:, cols_in_out] = integrated
                    x0 = (
                        np.array(self._states[p]["all"][:, next_state_col])
                        if shooting_type == Shooting.MULTIPLE
                        else integrated[:, -1]
                    )

            if not continuous:
                out._states[p]["all"][:, -1] = self._states[p]["all"][:, -1]

            # Dispatch the integrated values to all the keys
            for key in self.state_keys:
                out._states[p][key] = out._states[p]["all"][nlp.states[key].index, :]

            sum_states_len += out._states[p]["all"].shape[1]

        return out

    def interpolate(self, n_frames: Union[int, list, tuple]) -> Any:
        """
        Interpolate the states

        Parameters
        ----------
        n_frames: Union[int, list, tuple]
            If the value is an int, the Solution returns merges the phases,
            otherwise, it interpolates them independently

        Returns
        -------
        A Solution data structure with the states integrated. The controls are removed from this structure
        """

        out = self.copy(skip_data=True)

        t_all = []
        for p, data in enumerate(self._states):
            nlp = self.ocp.nlp[p]
            if nlp.ode_solver.is_direct_collocation and not self.recomputed_time_steps:
                time_offset = sum(out.phase_time[: p + 1])
                step_time = np.array(nlp.dynamics[0].step_time)
                dt = out.phase_time[p + 1] / nlp.ns
                t_tp = np.array([step_time * dt + s * dt + time_offset for s in range(nlp.ns)]).reshape(-1, 1)
                t_all.append(np.concatenate((t_tp, [[t_tp[-1, 0]]]))[:, 0])
            else:
                t_all.append(np.linspace(sum(out.phase_time[: p + 1]), sum(out.phase_time[: p + 2]), out.ns[p] + 1))

        if isinstance(n_frames, int):
            data_states, _, out.phase_time, out.ns = self._merge_phases(skip_controls=True)
            t_all = [np.concatenate((np.concatenate([_t[:-1] for _t in t_all]), [t_all[-1][-1]]))]

            n_frames = [n_frames]
            out.is_merged = True
        elif isinstance(n_frames, (list, tuple)) and len(n_frames) == len(self._states):
            data_states = self._states
        else:
            raise ValueError(
                "n_frames should either be a int to merge_phases phases "
                "or a list of int of the number of phases dimension"
            )

        out._states = []
        for _ in range(len(data_states)):
            out._states.append({})
        for p in range(len(data_states)):
            x_phase = data_states[p]["all"]
            n_elements = x_phase.shape[0]

            t_phase = t_all[p]
            t_phase, time_index = np.unique(t_phase, return_index=True)
            t_int = np.linspace(t_phase[0], t_phase[-1], n_frames[p])

            x_interpolate = np.ndarray((n_elements, n_frames[p]))
            for j in range(n_elements):
                s = sci_interp.splrep(t_phase, x_phase[j, time_index], k=1)
                x_interpolate[j, :] = sci_interp.splev(t_int, s)
            out._states[p]["all"] = x_interpolate

            offset = 0
            for key in data_states[p]:
                if key == "all":
                    continue
                n_elements = data_states[p][key].shape[0]
                out._states[p][key] = out._states[p]["all"][offset : offset + n_elements]
                offset += n_elements

        out.is_interpolated = True
        return out

    def merge_phases(self) -> Any:
        """
        Get a data structure where all the phases are merged into one

        Returns
        -------
        The new data structure with the phases merged
        """

        new = self.copy(skip_data=True)
        new.parameters = deepcopy(self.parameters)
        new._states, new._controls, new.phase_time, new.ns = self._merge_phases()
        new.is_merged = True
        return new

    def _merge_phases(self, skip_states: bool = False, skip_controls: bool = False, continuous: bool = True) -> tuple:
        """
        Actually performing the phase merging

        Parameters
        ----------
        skip_states: bool
            If the merge should ignore the states
        skip_controls: bool
            If the merge should ignore the controls
        continuous: bool
            If the last frame of each phase should be kept [False] or discard [True]

        Returns
        -------
        A tuple containing the new states, new controls, the recalculated phase time
        and the new number of shooting points
        """

        if self.is_merged:
            return deepcopy(self._states), deepcopy(self._controls), deepcopy(self.phase_time), deepcopy(self.ns)

        def _merge(data: list, is_control: bool) -> Union[list, dict]:
            """
            Merge the phases of a states or controls data structure

            Parameters
            ----------
            data: list
                The data to structure to merge the phases
            is_control: bool
                If the current data is a control

            Returns
            -------
            The data merged
            """

            if isinstance(data, dict):
                return data

            # Sanity check (all phases must contain the same keys with the same dimensions)
            keys = data[0].keys()
            sizes = [data[0][d].shape[0] for d in data[0]]
            for d in data:
                if d.keys() != keys or [d[key].shape[0] for key in d] != sizes:
                    raise RuntimeError("Program dimension must be coherent across phases to merge_phases them")

            data_out = [{}]
            for i, key in enumerate(keys):
                data_out[0][key] = np.ndarray((sizes[i], 0))

            add = 0 if is_control or continuous else 1
            for p in range(len(data)):
                d = data[p]
                for key in d:
                    if self.ocp.nlp[p].ode_solver.is_direct_collocation and not is_control:
                        steps = self.ocp.nlp[p].ode_solver.steps + 1
                        data_out[0][key] = np.concatenate(
                            (data_out[0][key], d[key][:, : self.ns[p] * steps + add]), axis=1
                        )
                    else:
                        data_out[0][key] = np.concatenate((data_out[0][key], d[key][:, : self.ns[p] + add]), axis=1)
            if add == 0:
                for key in data[-1]:
                    data_out[0][key] = np.concatenate((data_out[0][key], data[-1][key][:, -1][:, np.newaxis]), axis=1)

            return data_out

        if len(self._states) == 1:
            out_states = deepcopy(self._states)
        else:
            out_states = _merge(self.states, is_control=False) if not skip_states and self._states else None

        if len(self._controls) == 1:
            out_controls = deepcopy(self._controls)
        else:
            out_controls = _merge(self.controls, is_control=True) if not skip_controls and self._controls else None
        phase_time = [0] + [sum([self.phase_time[i + 1] for i in range(len(self.phase_time) - 1)])]
        ns = [sum(self.ns)]

        return out_states, out_controls, phase_time, ns

    def _complete_control(self):
        """
        Controls don't necessarily have dimensions that matches the states. This method aligns them
        """

        for p, nlp in enumerate(self.ocp.nlp):
            if nlp.control_type == ControlType.CONSTANT:
                for key in self._controls[p]:
                    self._controls[p][key] = np.concatenate(
                        (self._controls[p][key], np.nan * np.zeros((self._controls[p][key].shape[0], 1))), axis=1
                    )
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                pass
            else:
                raise NotImplementedError(f"ControlType {nlp.control_type} is not implemented  in _complete_control")
