"""
This script is reading and organizing the raw data results from Miller Optimal control problems into a nice DataFrame.
It requires the all the raw data to run the script.
"""
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import plotly
from plotly.subplots import make_subplots
import plotly.express as px

import biorbd
from bioptim import OptimalControlProgram

from utils import (
    stack_states,
    stack_controls,
    define_time,
    compute_error_single_shooting,
    compute_error_single_shooting_each_frame,
    my_traces,
    my_twokey_traces,
    add_annotation_letter,
    generate_windows_size,
    plot_all_dof,
    get_trans_and_rot_idx,
)

from enums import ResultFolders
from robot_leg import Models


class ResultsAnalyse:
    """
    This class is used to plot the results of the analysis.

    Attributes
    ----------
    df : pd.DataFrame
        The pandas dataframe with the results.
    df_path : str
        The path to the pandas dataframe.
    path_to_files : str
        The path to the folder containing the results.
    model_path : str
        The path to the model.
    path_to_figures : str
        The path to the folder containing the exported figures.
    path_to_data : str
        The path to the folder containing the exported data.

    Methods
    -------
    to_panda_dataframe(export: bool = True)
        Convert the data to a pandas dataframe.
    print
        Print results.
    plot_time_iter
        Plot the time evolution of the number of iterations.
    plot_integration_error
        Plot the integration error.
    plot_obj_values
        Plot the objective values.
    """

    def __init__(
            self,
            path_to_files: str,
            model_path: str,
            df_path: str = None,
            df: pd.DataFrame = None,
            ode_solvers: list = None,
            consistent_threshold: float = 10,
    ):

        self.path_to_files = path_to_files
        self.model_path = model_path
        self.model = biorbd.Model(self.model_path)

        # create a subfolder of self.path_to_files to export figures and data
        self.path_to_figures = f"{self.path_to_files}/figures"
        Path(self.path_to_figures).mkdir(parents=True, exist_ok=True)
        self.path_to_data = f"{self.path_to_files}/data"
        Path(self.path_to_data).mkdir(parents=True, exist_ok=True)

        self.consistent_threshold = consistent_threshold
        self.df_path = df_path
        self.df = df

        self.convergence_rate = pd.DataFrame(columns=["n_shooting", "convergence_rate", "ode_solver_defects", "grps"])
        self.ode_solvers = ode_solvers

    @classmethod
    def from_folder(
            cls,
            path_to_files: str,
            model_path: str,
            consistent_threshold: float = 10,
            export: bool = True,
    ):
        """
        Convert the data to a pandas dataframe.

        Parameters
        ----------
        path_to_files: str
            The path to the folder containing the results.
        model_path: str
            The path to the model.
        consistent_threshold: float
            The threshold to consider the OCP results.
        export : bool
            Export the dataframe as a pickle file.

        Returns
        -------
        df_results : pd.DataFrame
            The pandas dataframe with the results.
        """

        # open files
        files = os.listdir(path_to_files)
        files.sort()

        column_names = [
            "model_path",
            "irand",
            "extra_obj",
            "computation_time",
            "cost",
            "detailed_cost",
            "iterations",
            "status",
            "states" "controls",
            "parameters",
            "time",
            "dynamics_type",
            "q",
            "qdot",
            "q_integrated",
            "qdot_integrated",
            "tau",
            "n_shooting",
            "n_shooting_per_second",
            "n_theads",
            "grps",
            "grps_cat",
            "grp_number",
            "translation_error_traj",
            "rotation_error_traj",
            "index_10_deg_rmse",
        ]
        df_results = pd.DataFrame(columns=column_names)

        for i, file in enumerate(files):
            if file.endswith(".pckl"):
                print(file)
                p = Path(f"{path_to_files}/{file}")
                file_path = open(p, "rb")
                data = pickle.load(file_path)

                # _, sol = OptimalControlProgram.load(f"{self.path_to_files}/{p.stem}.bo")
                # DM to array
                data["filename"] = file
                data["tau"] = data["controls"]["tau"]

                data["cost"] = np.array(data["cost"])[0][0]
                # print(data["n_threads"])
                # compute error
                model = biorbd.Model(model_path)

                data["n_shooting_per_second"] = data["n_shooting"] / data["time"][-1]

                n_shooting = data["n_shooting"]
                q = data["q"]
                q_integrated = data["q_integrated"]
                # # print(data["q_integrated"].shape)

                (
                    data["translation_error"],
                    data["rotation_error"],
                ) = compute_error_single_shooting(
                    model=model,
                    n_shooting=n_shooting,
                    time=np.array(data["time"]),
                    q=q,
                    q_integrated=q_integrated,
                )

                (
                    data["translation_error_traj"],
                    data["rotation_error_traj"],
                ) = compute_error_single_shooting_each_frame(
                    model=model,
                    n_shooting=n_shooting,
                    time=np.array(data["time"]),
                    q=q,
                    q_integrated=q_integrated,
                )

                # to identify the point at which the consistency is sufficient
                idx = np.where(
                    data["rotation_error_traj"] > consistent_threshold
                )[0]
                data["consistent_threshold"] = idx[0] if idx.shape[0] != 0 else None

                data["ode_solver_defects"] = f"{data['ode_solver'].__str__()}_{data['defects_type'].value}"
                data[
                    "grps"
                ] = f"{data['ode_solver'].__str__()}_{data['defects_type'].value}_{n_shooting}"

                # remove element of the list[dict] data["detailed_cost"] if key name contains "ConstraintFcn"
                data["detailed_cost"] = [
                    {k: v for k, v in d.items() if "ConstraintFcn" not in d["name"]}
                    for d in data["detailed_cost"]
                ]
                data["detailed_cost"] = [d for d in data["detailed_cost"] if d]

                for i, cost in enumerate(data["detailed_cost"]):
                    data[f"cost{i}"] = cost["cost_value_weighted"]
                    data[f"cost{i}_details"] = cost

                df_dictionary = pd.DataFrame([data])
                df_results = pd.concat([df_results, df_dictionary], ignore_index=True)

        # set parameters of pandas to display all the columns of the pandas dataframe in the console
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        # don't return line with columns of pandas dataframe
        pd.set_option("display.expand_frame_repr", False)
        # display all the rows of the dataframe
        pd.set_option("display.max_rows", 20)

        # print(df_results[["dynamics_type", "n_shooting", "ode_solver", "translation_error", "rotation_error"]])
        # df_results[["dynamics_type", "ode_solver", "status", "translation_error", "rotation_error"]].to_csv(
        #     f"{self.path_to_files}/results.csv"
        # )

        # for each type of elements of grps, identify unique elements of grps
        df_results["grps_cat"] = pd.Categorical(df_results["grps"])
        df_results["grp_number"] = df_results["grps_cat"].cat.codes

        # sort the dataframe by the column by ode_solver_defects
        # rk4, rk8, irk explicit, irk implicit, collocation explicit, collocation implicit
        ode_solver_defects_list = [
            "RK4 5 steps_not_applicable",
            "RK8 2 steps_not_applicable",
            "IRK legendre 4_explicit",
            "IRK legendre 4_implicit",
            "COLLOCATION legendre 4_explicit",
            "COLLOCATION legendre 4_implicit",
        ]
        ode_solver_defects_list_updated = [cat for cat in ode_solver_defects_list if cat in df_results["ode_solver_defects"].unique()]
        df_results["ode_solver_defects"] = pd.Categorical(df_results["ode_solver_defects"], ode_solver_defects_list_updated)

        df_results.sort_values("ode_solver_defects", ascending=True, inplace=True)
        # reindex the dataframe
        df_results = df_results.reset_index(drop=True)

        # saves the dataframe
        if export:
            path_to_data = f"{path_to_files}/data"
            Path(path_to_data).mkdir(parents=True, exist_ok=True)
            df_path = f"{path_to_data}/Dataframe_results_metrics.pkl"
            df_results.to_pickle(df_path)

        return cls(
            path_to_files=path_to_files,
            model_path=model_path,
            consistent_threshold=consistent_threshold,
            df_path=df_path,
            df=df_results,
            ode_solvers=ode_solver_defects_list_updated,
        )

    def cluster(self, n_clusters: int = 2):
        """
        Clusters the OCPs according to their objective values

        Parameters
        ----------
        n_clusters: int
            The number of clusters to use
        """

        # Cluster the OCPs according to the objective values
        self.n_clusters = n_clusters
        self.df["cluster"] = None
        idx = np.where(self.df["status"] == 0)[0]
        df = self.df[self.df["status"] == 0]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
            df[["cost"]].values
        )
        for i, id in enumerate(idx):
            self.df.loc[id, "cluster"] = kmeans.labels_[i]

    def print(self):
        """
        Prints some info about the dataframe and convergence of OCPs
        """

        # Did everything converge ?
        a = len(self.df[self.df["status"] == 1])
        b = len(self.df)
        print(f"{a} / {b} did not converge to an optimal solutions")
        formulation = self.df["grps"].unique()

        for f in formulation:
            sub_df = self.df[self.df["grps"] == f]
            str_formulation = f.replace("_", " ").replace("-", " ").replace("\n", " ")
            a = len(sub_df[sub_df["status"] == 1])
            b = len(sub_df)
            print(
                f"{a} / {b} {str_formulation} did not converge to an optimal solutions"
            )

            data = dict(convergence_rate=1 - (a / b),
                        ode_solver_defects=sub_df["ode_solver_defects"].unique()[0],
                        n_shooting=sub_df["n_shooting"].unique()[0],
                        number_of_ocp=b,
                        number_of_convergence=a,
                        grps=f)
            df_dictionary = pd.DataFrame([data])
            self.convergence_rate = pd.concat([self.convergence_rate, df_dictionary], ignore_index=True)

            if "cluster" in self.df.keys():
                # print the number of element in each cluster
                for i in sub_df["cluster"].unique().tolist():
                    if i is not None:
                        sub_df_cluster = sub_df[sub_df["cluster"] == i]
                        a = len(sub_df_cluster)
                        b = len(sub_df)
                        print(f"{a} / {b} converge to cluster {i}")

        # sort the convergence rate dataframe
        self.convergence_rate["ode_solver_defects"] = pd.Categorical(self.convergence_rate["ode_solver_defects"], self.ode_solvers)
        self.convergence_rate.sort_values(by=["n_shooting", "ode_solver_defects"], ascending=True, inplace=True)
        # reindex the dataframe
        self.convergence_rate = self.convergence_rate.reset_index(drop=True)

    def animate(self, num: int = 0):
        """
        This method animates the motion with bioviz

        Parameters
        ----------
        num: int
        Number of the trial to be visualized
        """

        print(self.df["filename"].iloc[num])
        print(self.df["grps"].iloc[num])

        p = Path(self.df["model_path"].iloc[num])
        # verify if the path/file exists with pathlib
        model_path = self.model_path if not p.exists() else p.__str__()

        import bioviz
        biorbd_viz = bioviz.Viz(model_path,
                                show_now=False,
                                show_meshes=True,
                                show_global_center_of_mass=False,
                                show_gravity_vector=False,
                                show_floor=False,
                                show_segments_center_of_mass=False,
                                show_global_ref_frame=False,
                                show_local_ref_frame=False,
                                show_markers=False,
                                show_muscles=False,
                                show_wrappings=False,
                                background_color=(1, 1, 1),
                                mesh_opacity=0.97,
                                )

        biorbd_viz.resize(600, 900)

        # Position camera
        biorbd_viz.set_camera_position(-8.782458942185185, 0.486269131372712, 4.362010279585766)
        biorbd_viz.set_camera_roll(90)
        biorbd_viz.set_camera_zoom(0.308185240948253)
        biorbd_viz.set_camera_focus_point(1.624007185850899, 0.009961251074366406, 1.940316420941989)

        # print("roll")
        # print(biorbd_viz.get_camera_roll())
        # print("zoom")
        # print(biorbd_viz.get_camera_zoom())
        # print("position")
        # print(biorbd_viz.get_camera_position())
        # print("get_camera_focus_point")
        # print(biorbd_viz.get_camera_focus_point())

        q = self.df["q"].iloc[num]
        biorbd_viz.load_movement(q)
        biorbd_viz.exec()

    def plot_time_iter(
            self, show: bool = True, export: bool = True, time_unit: str = "s", export_suffix : str = None):
        """
        This function plots the time and number of iterations need to make the OCP converge

        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        time_unit : str
            The time unit of the figure.
        """

        # dyn = [i for i in df_results["grps"].unique().tolist() if "COLLOCATION" in i and "legendre" in i]
        dyn = self.df["grps"].unique().tolist()
        grps = dyn

        fig = make_subplots(rows=1, cols=2)

        # select only the one who converged
        df_results = self.df[self.df["status"] == 0]

        if time_unit == "min":
            df_results["computation_time"] = df_results["computation_time"] / 60

        fig = my_traces(
            fig,
            dyn,
            grps,
            df_results,
            "computation_time",
            1,
            1,
            "time (s)",
            ylog=False,
        )
        fig = my_traces(
            fig,
            dyn,
            grps,
            df_results,
            "iterations",
            1,
            2,
            r"$\text{iterations}$",
            ylog=False,
        )

        fig.update_layout(
            height=800,
            width=1500,
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            legend=dict(
                title_font_family="Times New Roman",
                font=dict(family="Times New Roman", color="black", size=11),
                orientation="h",
                xanchor="center",
                x=0.5,
                y=-0.05,
            ),
            font=dict(
                size=12,
                family="Times New Roman",
            ),
            yaxis=dict(color="black"),
            template="simple_white",
            boxgap=0.2,
        )
        fig = add_annotation_letter(fig, "A", x=0.01, y=0.99, on_paper=True)
        fig = add_annotation_letter(fig, "B", x=0.56, y=0.99, on_paper=True)

        fig.update_yaxes(
            row=1,
            col=1,
            tickformat=".1f",
        )
        fig.update_yaxes(
            row=1,
            col=2,
            tickformat=".0f",
        )

        if show:
            fig.show()
        if export:
            format_type = ["png", "pdf", "svg", "eps"]
            for f in format_type:
                fig.write_image(self.path_to_figures + f"/analyse_time_iter{export_suffix}." + f)
            fig.write_html(
                self.path_to_figures + f"/analyse_time_iter{export_suffix}.html", include_mathjax="cdn"
            )

    def plot_integration_frame_to_frame_error(
            self, show: bool = True,
            export: bool = True,
            until_consistent: bool = False,
            export_suffix : str = None
    ):
        """
        This function plots the time and number of iterations need to make the OCP converge

        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        until_consistent : bool
            plot the curves until it respect the consistency we wanted
        """

        # dyn = [i for i in self.df["grps"].unique().tolist() if "COLLOCATION" in i and "legendre" in i]
        dyn = self.df["grps"].unique().tolist()
        grps = dyn

        fig = make_subplots(
            rows=1, cols=2, subplot_titles=["translation error", "rotation error"]
        )
        # update the font size of the subplot_titles
        for i in fig["layout"]["annotations"]:
            i["font"] = dict(size=18)

        # select only the one who converged
        df_results = self.df[self.df["status"] == 0]

        for _, row in df_results.iterrows():
            idx_end = (
                int(row.consistent_threshold)
                if until_consistent
                else int(len(row.time))
            )
            print(idx_end)
            time = row.time[:idx_end]
            y1 = row["translation_error_traj"][:idx_end]
            y2 = row["rotation_error_traj"][:idx_end]

            fig.add_scatter(
                x=time,
                y=y1,
                mode="lines",
                marker=dict(
                    size=1,
                    color=px.colors.qualitative.D3[row.grp_number],
                    line=dict(width=0.05, color="DarkSlateGrey"),
                ),
                name=row.grps if row.irand == 0 else None,
                legendgroup=row.grps,
                showlegend=True if row.irand == 0 else False,
            )

            fig.add_scatter(
                x=time,
                y=y2,
                mode="lines",
                marker=dict(
                    size=1,
                    color=px.colors.qualitative.D3[row.grp_number],
                    line=dict(width=0.05, color="DarkSlateGrey"),
                ),
                name=row.grps if row.irand == 0 else None,
                legendgroup=row.grps,
                showlegend=False,
                row=1,
                col=2,
            )

        fig.update_layout(
            height=800,
            width=1500,
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            legend=dict(
                title_font_family="Times New Roman",
                font=dict(family="Times New Roman", color="black", size=11),
                orientation="h",
                xanchor="center",
                x=0.5,
                y=-0.05,
            ),
            font=dict(
                size=12,
                family="Times New Roman",
            ),
            yaxis=dict(color="black"),
            template="simple_white",
            boxgap=0.2,
        )
        fig = add_annotation_letter(fig, "A", x=0.01, y=0.99, on_paper=True)
        fig = add_annotation_letter(fig, "B", x=0.56, y=0.99, on_paper=True)

        fig.update_yaxes(
            row=1,
            col=1,
            type="log",
        )
        fig.update_yaxes(
            row=1,
            col=2,
            type="log",
        )

        if show:
            fig.show()
        if export:
            format_type = ["png", "pdf", "svg", "eps"]
            for f in format_type:
                fig.write_image(self.path_to_figures + f"/analyse_integration_each_frame{export_suffix}." + f)
            fig.write_html(
                self.path_to_figures + f"/analyse_integration_each_frame{export_suffix}.html",
                include_mathjax="cdn",
            )

    def plot_integration_final_error(self, show: bool = True, export: bool = True, export_suffix : str = None):
        """
        This function plots the time and number of iterations need to make the OCP converge

        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        """

        # dyn = [i for i in self.df["grps"].unique().tolist() if "COLLOCATION" in i and "legendre" in i]
        dyn = self.df["grps"].unique().tolist()
        grps = dyn

        fig = make_subplots(rows=1, cols=1)

        # select only the one who converged
        df_results = self.df[self.df["status"] == 0]

        fig = my_traces(
            fig,
            dyn,
            grps,
            df_results,
            "rotation_error",
            row=1,
            col=1,
            ylabel="degrees",
            ylog=True,
        )

        fig.update_layout(
            height=800,
            width=1500,
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            legend=dict(
                title_font_family="Times New Roman",
                font=dict(family="Times New Roman", color="black", size=11),
                orientation="h",
                xanchor="center",
                x=0.5,
                y=-0.05,
            ),
            font=dict(
                size=12,
                family="Times New Roman",
            ),
            yaxis=dict(color="black"),
            template="simple_white",
            boxgap=0.2,
        )
        fig = add_annotation_letter(fig, "A", x=0.01, y=0.99, on_paper=True)

        fig.update_yaxes(
            row=1,
            col=1,
        )
        fig.update_yaxes(
            row=1,
            col=2,
        )

        if show:
            fig.show()
        if export:
            format_type = ["png", "pdf", "svg", "eps"]
            for f in format_type:
                fig.write_image(self.path_to_figures + f"/analyse_integration{export_suffix}." + f)
            fig.write_html(
                self.path_to_figures + f"/analyse_integration{export_suffix}.html",
                include_mathjax="cdn",
            )

    def plot_obj_values(self, show: bool = True, export: bool = True, export_suffix : str = None):
        """
        This function plots the time and number of iterations need to make the OCP converge

        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        """

        # dyn = [i for i in self.df["grps"].unique().tolist() if "COLLOCATION" in i and "legendre" in i]
        dyn = self.df["grps"].unique().tolist()
        grps = dyn

        fig = make_subplots(rows=1, cols=1)

        # select only the one who converged
        df_results = self.df[self.df["status"] == 0]

        fig = my_traces(
            fig,
            dyn,
            grps,
            df_results,
            key="cost",
            row=1,
            col=1,
            ylabel="objective value",
        )

        fig.update_layout(
            height=800,
            width=1500,
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            legend=dict(
                title_font_family="Times New Roman",
                font=dict(family="Times New Roman", color="black", size=11),
                orientation="h",
                xanchor="center",
                x=0.5,
                y=-0.05,
            ),
            font=dict(
                size=12,
                family="Times New Roman",
            ),
            yaxis=dict(color="black"),
            template="simple_white",
            boxgap=0.2,
        )

        fig.update_yaxes(
            row=1,
            col=1,
        )

        if show:
            fig.show()
        if export:
            format_type = ["png", "pdf", "svg", "eps"]
            for f in format_type:
                fig.write_image(self.path_to_figures + f"/analyse_obj{export_suffix}." + f)
            fig.write_html(
                self.path_to_figures + f"/analyse_obj{export_suffix}.html", include_mathjax="cdn"
            )

    def _plot_2_keys(
            self,
            key_x: str,
            key_y: str,
            x_label: str,
            y_label: str,
            x_log: bool = False,
            y_log: bool = False,
            show: bool = True,
            export: bool = True,
            export_suffix : str = None,
    ):
        """
        This function plots the time and number of iterations need to make the OCP converge

        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        """

        # dyn = [i for i in self.df["grps"].unique().tolist() if "COLLOCATION" in i and "legendre" in i]
        dyn = self.df["grps"].unique().tolist()
        grps = dyn

        fig = make_subplots(rows=1, cols=1)

        # select only the one who converged
        df_results = self.df[self.df["status"] == 0]

        fig = my_twokey_traces(
            fig,
            dyn,
            grps,
            df_results,
            key_x=key_x,
            key_y=key_y,
            row=1,
            col=1,
            ylabel=y_label,
            xlabel=x_label,
            ylog=y_log,
            xlog=x_log,
        )

        fig.update_layout(
            height=800,
            width=1500,
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            legend=dict(
                title_font_family="Times New Roman",
                font=dict(family="Times New Roman", color="black", size=11),
                orientation="h",
                xanchor="center",
                x=0.5,
                y=-0.1,
            ),
            font=dict(
                size=12,
                family="Times New Roman",
            ),
            yaxis=dict(color="black"),
            xaxis=dict(color="black"),
            template="simple_white",
            # show grid with black lines on ticks
            xaxis_showgrid=True,
            yaxis_showgrid=True,
        )

        fig.update_yaxes(
            row=1,
            col=1,
        )

        return fig

    def plot_cost_vs_consistency(self, show: bool = True, export: bool = True, export_suffix : str = None):
        """
        This function plots the time and number of iterations need to make the OCP converge

        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        """

        fig = self._plot_2_keys(
            key_x="cost",
            key_y="rotation_error",
            x_label="objective function value",
            y_label="final rotation RMSE (degree)",
            y_log=True,
        )

        if show:
            fig.show()
        if export:
            format_type = ["png", "pdf", "svg", "eps"]
            for f in format_type:
                fig.write_image(self.path_to_figures + f"/analyse_time_vs_obj{export_suffix}." + f)
            fig.write_html(
                self.path_to_figures + f"/analyse_time_vs_obj{export_suffix}.html", include_mathjax="cdn"
            )

    def plot_time_vs_consistency(self, show: bool = True, export: bool = True, export_suffix : str = None):
        """
        This function plots the time and number of iterations need to make the OCP converge

        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        """

        fig = self._plot_2_keys(
            key_x="computation_time",
            key_y="rotation_error",
            x_label="time (s)",
            y_label="final rotation RMSE (degree)",
            y_log=True,
        )

        if show:
            fig.show()
        if export:
            format_type = ["png", "pdf", "svg", "eps"]
            for f in format_type:
                fig.write_image(self.path_to_figures + f"/analyse_time_vs_obj{export_suffix}." + f)
            fig.write_html(
                self.path_to_figures + f"/analyse_time_vs_obj{export_suffix}.html", include_mathjax="cdn"
            )

    def plot_time_vs_obj(self, show: bool = True, export: bool = True, export_suffix : str = None):
        """
        This function plots the time and number of iterations need to make the OCP converge

        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        """

        fig = self._plot_2_keys(
            key_x="computation_time",
            key_y="cost",
            x_label="time (s)",
            y_label="objective value",
        )

        if show:
            fig.show()
        if export:
            format_type = ["png", "pdf", "svg", "eps"]
            for f in format_type:
                fig.write_image(self.path_to_figures + f"/analyse_time_vs_obj{export_suffix}." + f)
            fig.write_html(
                self.path_to_figures + f"/analyse_time_vs_obj{export_suffix}.html", include_mathjax="cdn"
            )

    def plot_detailed_obj_values(self, show: bool = True, export: bool = True, export_suffix : str = None):
        """
        This function plots the time and number of iterations need to make the OCP converge
        # todo
        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        """

        # dyn = [i for i in self.df["grps"].unique().tolist() if "COLLOCATION" in i and "legendre" in i]
        dyn = self.df["grps"].unique().tolist()
        grps = dyn

        # select only the one who converged
        df_results = self.df[self.df["status"] == 0]

        nb_costs = df_results["detailed_cost"][0].__len__()
        rows, cols = generate_windows_size(nb_costs)

        # indices of rows and cols for each axe of the subplot
        idx_rows, idx_cols = np.unravel_index([i for i in range(nb_costs)], (rows, cols))
        idx_rows += 1
        idx_cols += 1

        titles = []
        for i in range(nb_costs):
            name = df_results[f"cost{i}_details"][0]["name"]
            # key = " " if df_results[f"cost{i}_details"][0]["params"]["key"] is not None else ""
            # if param is not empty, key is ""
            param = df_results[f"cost{i}_details"][0]["params"]
            key = " " + param["key"] if param else ""

            derivative = "delta " if df_results[f"cost{i}_details"][0]["derivative"] else ""
            titles.append(f"{derivative}{name}{key}")

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

        for i in range(nb_costs):

            fig = my_traces(
                fig,
                dyn,
                grps,
                df_results,
                key=f"cost{i}",
                row=idx_rows[i],
                col=idx_cols[i],
                # title_str=df_results[f"cost{i}_name"][0] + " " + key,
                ylabel="objective value",
            )

        fig.update_layout(
            height=800,
            width=1500,
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            legend=dict(
                title_font_family="Times New Roman",
                font=dict(family="Times New Roman", color="black", size=11),
                orientation="h",
                xanchor="center",
                x=0.5,
                y=-0.05,
            ),
            font=dict(
                size=12,
                family="Times New Roman",
            ),
            yaxis=dict(color="black"),
            template="simple_white",
            boxgap=0.2,
        )

        fig.update_yaxes(
            row=1,
            col=1,
        )

        if show:
            fig.show()
        if export:
            format_type = ["png", "pdf", "svg", "eps"]
            for f in format_type:
                fig.write_image(self.path_to_figures + f"/analyse_detailed_obj{export_suffix}." + f)
            fig.write_html(
                self.path_to_figures + f"/analyse_obj{export_suffix}.html", include_mathjax="cdn"
            )

    def plot_state(
            self,
            key: str = None,
            show: bool = True,
            export: bool = True,
            label_dofs: list[str] = None,
            row_col: tuple[int, int] = None,
            ylabel_rotations: str = "q",
            ylabel_translations: str = "q",
            xlabel: str = "Time (s)",
            until_consistent: bool = False,
            export_suffix : str = None,
    ) -> plotly.graph_objects.Figure:
        """
        This function plots generalized coordinates of each OCPs

        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        label_dofs : list[str]
            List of labels for the degrees of freedom.
        row_col : tuple[int, int]
            Row and column of the subplot.
        ylabel_rotations : str
            Label for the y-axis of the rotations.
        ylabel_translations : str
            Label for the y-axis for translations.
        xlabel : str
            Label for the x-axis.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Figure object.
        """
        if "tau" in key:
            nq = 9
            list_dof = [dof.to_string() for dof in self.model.nameDof()][6:]
        else:
            nq = self.model.nbQ()
            list_dof = [dof.to_string() for dof in self.model.nameDof()]

        rows, cols = generate_windows_size(nq) if row_col is None else row_col

        # indices of rows and cols for each axe of the subplot
        idx_rows, idx_cols = np.unravel_index([i for i in range(nq)], (rows, cols))
        idx_rows += 1
        idx_cols += 1

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=label_dofs if not label_dofs is None else list_dof,
            vertical_spacing=0.05,
            shared_xaxes=True,
        )
        # update the font size of the subplot_titles
        for i in fig["layout"]["annotations"]:
            i["font"] = dict(size=18)

        # select only the one who converged
        df_results = self.df[self.df["status"] == 0]

        # handle translations and rotations
        trans_idx, rot_idx = get_trans_and_rot_idx(self.model)

        fig = plot_all_dof(
            fig,
            key,
            df_results,
            list_dof,
            idx_rows,
            idx_cols,
            trans_idx,
            rot_idx,
            until_consistent=until_consistent,
        )

        for i in range(1, cols + 1):
            fig.update_xaxes(row=rows, col=i, title=xlabel)

        # handle translations and rotations
        trans_idx, rot_idx = get_trans_and_rot_idx(self.model)

        if "tau" in key:
            trans_idx = []
            rot_idx = rot_idx[:-6]

        for idx in trans_idx:
            fig.update_yaxes(
                row=idx_rows[idx], col=idx_cols[idx], title=ylabel_translations
            )
        for idx in rot_idx:
            fig.update_yaxes(
                row=idx_rows[idx], col=idx_cols[idx], title=ylabel_rotations
            )

        if show:
            fig.show()
        if export:
            format_type = ["png", "pdf", "svg", "eps"]
            for f in format_type:
                fig.write_image(self.path_to_figures + f"/analyse_{key}{export_suffix}." + f)
            fig.write_html(
                self.path_to_figures + f"/analyse_{key}{export_suffix}.html", include_mathjax="cdn"
            )

        return fig

    def analyse(self, show=True, export=True, cluster_analyse=False):
        self.print()
        self.plot_time_iter(show=show, export=export, time_unit="min")
        # self.plot_obj_values(show=show, export=export)
        # # self.plot_detailed_obj_values(show=show, export=export)
        self.plot_time_vs_obj(show=show, export=export)
        self.plot_time_vs_consistency(show=show, export=export)
        self.plot_cost_vs_consistency(show=show, export=export)
        # self.plot_integration_frame_to_frame_error(
        #     show=show, export=export, until_consistent=False
        # )
        # self.plot_integration_final_error(show=show, export=export)
        # self.plot_state(
        #     key="q", show=show, export=export, row_col=(5, 3), until_consistent=False
        # )
        # self.plot_state(
        #     key="q_integrated",
        #     show=show,
        #     export=export,
        #     row_col=(5, 3),
        #     until_consistent=False,
        # )

    def cluster_analyse(self, show=True, animate=False):
        self.cluster(n_clusters=2)
        self.print()
        export = True

        for i in range(self.n_clusters):
            export_suffix = f"_cluster{i}"
            cluster_results = ResultsAnalyse(
                path_to_files=self.path_to_files,
                model_path=self.model_path,
                df=self.df[self.df["cluster"] == i]
            )
            if animate:
                cluster_results.animate()
            # cluster_results.plot_time_iter(show=show, export=export, time_unit="min", export_suffix=export_suffix)
            # cluster_results.plot_obj_values(show=show, export=export, export_suffix=export_suffix)
            # cluster_results.plot_detailed_obj_values(show=show, export=export, export_suffix=export_suffix)

            cluster_results.plot_time_vs_obj(show=show, export=export, export_suffix=export_suffix)
            cluster_results.plot_time_vs_consistency(show=show, export=export, export_suffix=export_suffix)
            cluster_results.plot_cost_vs_consistency(show=show, export=export, export_suffix=export_suffix)

            # cluster_results.plot_integration_frame_to_frame_error(
            #     show=show, export=export, until_consistent=False, export_suffix=export_suffix)
            # cluster_results.plot_integration_final_error(show=show, export=export, export_suffix=export_suffix)
            # cluster_results.plot_state(
            #     key="q", show=show, export=export, row_col=(5, 3), until_consistent=False, export_suffix=export_suffix)
            # cluster_results.plot_state(
            #     key="q_integrated",
            #     show=show,
            #     export=export,
            #     row_col=(5, 3),
            #     until_consistent=False,
            #     export_suffix=export_suffix,
            # )


if __name__ == "__main__":

    # results = ResultsAnalyse.from_folder(
    #     model_path=Models.LEG.value,
    #     path_to_files=ResultFolders.LEG.value,
    #     export=True,
    # )
    # results.analyse(
    #     show=True,
    #     export=True,
    # )
    results = ResultsAnalyse.from_folder(
        model_path=Models.ARM.value,
        path_to_files=ResultFolders.ARM.value,
        export=True,
    )
    # results.animate(num=5)
    results.analyse(
        show=True,
        export=True,
        cluster_analyse=True,
    )
    results.cluster_analyse(
        show=True,
    )

    results = ResultsAnalyse.from_folder(
            model_path=Models.ACROBAT.value,
            path_to_files=ResultFolders.ACROBAT.value,
            export=True,
        )
    results.analyse(
        show=True,
        export=True,
    )
