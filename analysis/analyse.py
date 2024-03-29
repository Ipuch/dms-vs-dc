"""
This script is reading and organizing the raw data results from Miller Optimal control problems into a nice DataFrame.
It requires the all the raw data to run the script.
"""
import os
from typing import List
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from plotly import graph_objects as go
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
from transcriptions import Models


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
            colors: dict = None,
    ):

        self.path_to_files = path_to_files
        self.model_path = model_path
        # self.model = biorbd.BioModel(self.model_path)

        # create a subfolder of self.path_to_files to export figures and data
        self.path_to_figures = f"{self.path_to_files}/figures"
        Path(self.path_to_figures).mkdir(parents=True, exist_ok=True)
        self.path_to_data = f"{self.path_to_files}/data"
        Path(self.path_to_data).mkdir(parents=True, exist_ok=True)

        self.consistent_threshold = consistent_threshold
        self.df_path = df_path
        self.df = df

        self.ode_solvers = ode_solvers
        self.colors = colors

        self.convergence_rate = pd.DataFrame(columns=["n_shooting", "convergence_rate", "ode_solver_defects", "grps"])
        self.print()

        self.near_optimal = pd.DataFrame(
            columns=[
                "n_shooting",
                "ode_solver_defects",
                "grps",
                "number of ocp",
                "number of near optimal ocp",
                "percent of near optimal ocp",
            ]
        )
        self.compute_near_optimality()

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
            if file.endswith(".pckl") and "RK8" not in file:
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
                model_path = model_path if isinstance(model_path, str) else model_path[0]
                model = biorbd.Model(model_path)

                df_results["computation_time_per_shooting"] = df_results["computation_time"] / df_results["n_shooting"]
                df_results["computation_time_per_shooting_per_var"] = (
                        df_results["computation_time"]
                        / df_results["n_shooting"]
                        / (model.nbQ() + model.nbQdot() + model.nbGeneralizedTorque())
                )

                data["n_shooting_per_second"] = data["n_shooting"] / data["time"][-1]

                n_shooting = data["n_shooting"]
                q = data["q"]
                q_integrated = data["q_integrated"]
                # # print(data["q_integrated"].shape)

                (data["translation_error"], data["rotation_error"],) = compute_error_single_shooting(
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
                idx = np.where(data["rotation_error_traj"] > consistent_threshold)[0]
                data["consistent_threshold"] = idx[0] if idx.shape[0] != 0 else None

                # errors per second
                data["rotation_error_per_second"] = data["rotation_error"] / data["time"][-1]
                data["rotation_error_per_second_per_velocity_max"] = (
                        data["rotation_error"]
                        / data["time"][-1]
                        / data["qdot"].max()
                        / (model.nbQ() + model.nbQdot() + model.nbGeneralizedTorque())
                )

                # labels and groups with ode solvers
                data["ode_solver_defects"] = f"{data['ode_solver'].__str__()}_{data['defects_type'].value}"
                # clean ode_solver_defects to display a nice label
                data["ode_solver_defects_labels"] = data["ode_solver_defects"].replace("_not_applicable", "").replace(
                    "_", " ").replace(" legendre 4", "").replace("RK4", "ERK4").replace("RK8", "ERK8")

                # replace labels for ode solvers
                if data["ode_solver_defects_labels"] == "ERK4 5 steps":
                    data["ode_solver_defects_labels"] = r'$\text{ERK}$'
                elif data["ode_solver_defects_labels"] == "ERK8":
                    data["ode_solver_defects_labels"] = r'$\text{ERK8}$'
                elif data["ode_solver_defects_labels"] == "IRK implicit":
                    data["ode_solver_defects_labels"] = r'$\text{IRK}^{\text{ID}}$'
                elif data["ode_solver_defects_labels"] == "IRK explicit":
                    data["ode_solver_defects_labels"] = r'$\text{IRK}^{\text{FD}}$'
                elif data["ode_solver_defects_labels"] == "COLLOCATION implicit":
                    data["ode_solver_defects_labels"] = r'$\text{DC}^{\text{ID}}$'
                elif data["ode_solver_defects_labels"] == "COLLOCATION explicit":
                    data["ode_solver_defects_labels"] = r'$\text{DC}^{\text{FD}}$'

                data["grps"] = f"{data['ode_solver'].__str__()}_{data['defects_type'].value}_{n_shooting}"

                # remove element of the list[dict] data["detailed_cost"] if key name contains "ConstraintFcn"
                data["detailed_cost"] = [
                    {k: v for k, v in d.items() if "ConstraintFcn" not in d["name"]} for d in data["detailed_cost"]
                ]
                data["detailed_cost"] = [d for d in data["detailed_cost"] if d]

                for i, cost in enumerate(data["detailed_cost"]):
                    data[f"cost{i}"] = cost["cost_value_weighted"]
                    data[f"cost{i}_details"] = cost

                df_dictionary = pd.DataFrame([data])
                df_results = pd.concat([df_results, df_dictionary], ignore_index=True)

        # sort the dataframe by the column by ode_solver_defects
        # rk4, rk8, irk explicit, irk implicit, collocation explicit, collocation implicit
        ode_solver_defects_list = [
            "RK4 5 steps_not_applicable",
            "IRK legendre 4_explicit",
            "IRK legendre 4_implicit",
            "COLLOCATION legendre 4_explicit",
            "COLLOCATION legendre 4_implicit",
        ]
        colors = {ode: px.colors.qualitative.D3[i] for i, ode in enumerate(ode_solver_defects_list)}

        ode_solver_defects_list_updated = [
            cat for cat in ode_solver_defects_list if cat in df_results["ode_solver_defects"].unique()
        ]
        df_results["ode_solver_defects"] = pd.Categorical(
            df_results["ode_solver_defects"], ode_solver_defects_list_updated
        )

        df_results.sort_values("ode_solver_defects", ascending=True, inplace=True)
        # reindex the dataframe
        df_results = df_results.reset_index(drop=True)

        near_optimal_magnitude = 1.5

        # the ones that converges
        df_results_converged = df_results[df_results["status"] == 0]
        # find the global minimum cost whatever the ode_solver_defects
        min_cost = df_results_converged["cost"].min()
        # find the costs that are within 15% of the global minimum cost
        min_cost_15 = min_cost * near_optimal_magnitude
        # find the index of the costs that are within 15% of the global minimum cost
        idx_min_cost_15 = df_results["cost"] < min_cost_15
        # set a new argument "near_optimal" to True else False if the cost is within 15% of the global minimum cost
        df_results["near_optimal"] = False
        df_results.loc[idx_min_cost_15, "near_optimal"] = True

        # find the global minimum cost for each ode_solver_defects
        for ode_solver_defects in df_results["ode_solver_defects"].unique():
            idx = df_results["ode_solver_defects"] == ode_solver_defects
            min_cost = df_results.loc[idx, "cost"].min()
            # find the costs that are within 15% of the global minimum cost
            min_cost_15 = min_cost * near_optimal_magnitude
            # find the index of the costs that are within 15% of the global minimum cost
            idx_min_cost_15 = df_results.loc[idx, "cost"] < min_cost_15
            # set a new argument "near_optimal" to True else False if the cost is within 15% of the global minimum cost
            df_results.loc[idx, "near_optimal_ode"] = False
            df_results.loc[idx & idx_min_cost_15, "near_optimal_ode"] = True

        # set parameters of pandas to display all the columns of the pandas dataframe in the console
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        # don't return line with columns of pandas dataframe
        pd.set_option("display.expand_frame_repr", False)
        # display all the rows of the dataframe
        pd.set_option("display.max_rows", 20)

        # for each type of elements of grps, identify unique elements of grps
        df_results["grps_cat"] = pd.Categorical(df_results["grps"])
        df_results["grp_number"] = df_results["grps_cat"].cat.codes

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
            colors=colors,
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
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df[["cost"]].values)
        for i, id in enumerate(idx):
            self.df.loc[id, "cluster"] = kmeans.labels_[i]

    def print(self):
        """
        Prints some info about the dataframe and convergence of OCPs
        """
        print("####################")
        print("BioModel path: ", self.model_path)
        print("####################")
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
            print(f"{a} / {b} {str_formulation} did not converge to an optimal solutions")

            data = dict(
                convergence_rate=1 - (a / b),
                ode_solver_defects=sub_df["ode_solver_defects"].unique()[0],
                n_shooting=sub_df["n_shooting"].unique()[0],
                number_of_ocp=b,
                number_of_convergence=a,
                grps=f,
                median=round(sub_df["computation_time"].median(), 2),
            )
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
        self.convergence_rate["ode_solver_defects"] = pd.Categorical(
            self.convergence_rate["ode_solver_defects"], self.ode_solvers
        )
        self.convergence_rate.sort_values(by=["n_shooting", "ode_solver_defects"], ascending=True, inplace=True)
        # reindex the dataframe
        self.convergence_rate = self.convergence_rate.reset_index(drop=True)

        print("####################")

        # compare implicit and explicit time medians
        sub_df = self.df[self.df["ode_solver_defects_labels"] == r'$\text{DC}^{\text{ID}}$']
        sub_df = sub_df[sub_df["status"] == 0]
        median_DCID = sub_df["computation_time"].median()
        max_DCID = sub_df["computation_time"].max()
        min_DCID = sub_df["computation_time"].min()

        sub_df = self.df[self.df["ode_solver_defects_labels"] == r'$\text{IRK}^{\text{ID}}$']
        sub_df = sub_df[sub_df["status"] == 0]
        median_IRKID = sub_df["computation_time"].median()
        max_IRKID = sub_df["computation_time"].max()
        min_IRKID = sub_df["computation_time"].min()

        sub_df = self.df[self.df["ode_solver_defects_labels"] == r'$\text{IRK}^{\text{FD}}$']
        sub_df = sub_df[sub_df["status"] == 0]
        median_IRKFD = sub_df["computation_time"].median()
        max_IRKFD = sub_df["computation_time"].max()
        min_IRKFD = sub_df["computation_time"].min()

        sub_df = self.df[self.df["ode_solver_defects_labels"] == r'$\text{DC}^{\text{FD}}$']
        sub_df = sub_df[sub_df["status"] == 0]
        median_DCFD = sub_df["computation_time"].median()
        max_DCFD = sub_df["computation_time"].max()
        min_DCFD = sub_df["computation_time"].min()

        print(f"median computation time for DCID: {median_DCID}, (min: {min_DCID}, max: {max_DCID})")
        print(f"median computation time for DCFD: {median_DCFD}, (min: {min_DCFD}, max: {max_DCFD})")
        # print ratio
        print(f"ratio DCID/DCFD: {median_DCID/median_DCFD}", f"ratio DCFD/DCID: {median_DCFD/median_DCID}")

        print(f"median computation time for IRKID: {median_IRKID}, (min: {min_IRKID}, max: {max_IRKID})")
        print(f"median computation time for IRKFD: {median_IRKFD}, (min: {min_IRKFD}, max: {max_IRKFD})")
        # print ratio
        print(f"ratio IRKID/IRKFD: {median_IRKID/median_IRKFD}", f"ratio IRKFD/IRKID: {median_IRKFD/median_IRKID}")

        print("####################")

        # fastest pb over slowest
        sub_df = self.df[self.df["status"] == 0]
        # the faster
        print("Fastest OCP", sub_df[sub_df["computation_time"] == sub_df["computation_time"].min()])
        # the slower
        print("Slowest OCP", sub_df[sub_df["computation_time"] == sub_df["computation_time"].max()])
        # the ratio
        print(f"ratio fastest pb over slowest: {sub_df['computation_time'].max()/sub_df['computation_time'].min()}")


    def plot_convergence_rate(self, show: bool = True, export: bool = True, export_suffix: str = None):
        """
        This function plots the number of problem that converged for each ode_solver and each number of nodes

        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        """

        # set the n_shooting column as categorical
        df = self.convergence_rate.copy()
        # n_shooting as str
        df["n_shooting"] = df["n_shooting"].astype(str)

        fig = px.histogram(
            df, x="n_shooting", y="convergence_rate", color="ode_solver_defects", barmode="group", height=400
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="black",
        )
        # sh

        # Update axis
        fig.update_xaxes(title_text="Number of nodes")
        fig.update_yaxes(title_text="Convergence rate (%)")

        # display horinzontal line grid
        fig.update_yaxes(showgrid=True, gridwidth=5)

        # set the colors of the bars with px.colors.qualitative.D3 for each ode_solver
        for i, ode_solver in enumerate(self.convergence_rate["ode_solver_defects"].unique()):
            fig.data[i].marker.color = self.colors[ode_solver]

        # bars are transparent a bit
        for i in range(len(fig.data)):
            fig.data[i].marker.opacity = 0.9

        # contours of bars are black
        for i in range(len(fig.data)):
            fig.data[i].marker.line.color = "black"
            fig.data[i].marker.line.width = 1

        if show:
            fig.show()
        if export:
            format_type = ["png", "pdf", "svg", "eps"]
            for f in format_type:
                fig.write_image(self.path_to_figures + f"/analyse_convergence_rate{export_suffix}." + f)
            fig.write_html(
                self.path_to_figures + f"/analyse_convergence_rate{export_suffix}.html", include_mathjax="cdn"
            )

    def compute_near_optimality(self):
        """
        This function fills the dictionnay self.near_optimal with the number of near optimal OCPs for each formulation
        """

        # the ones that converges
        df_results_converged = self.df[self.df["status"] == 0]
        # find the global minimum cost whatever the ode_solver_defects
        min_cost = df_results_converged["cost"].min()

        # compute the number of near optimal OCPs for each formulation
        for ode in self.df["ode_solver_defects"].unique():

            sub_df = self.df[self.df["ode_solver_defects"] == ode]
            # only the ones that converged
            sub_df = sub_df[sub_df["status"] == 0]

            # check if the sub_df is empty
            if len(sub_df) == 0:
                continue

            # compute cumulative near optimality
            nb_true_list = []
            for ii in range(0, 200):
                near_optimal_magnitude = 1 + ii * 0.01
                cur_cost_threshold = min_cost * near_optimal_magnitude
                idx_min_cost_threshold = sub_df["cost"] <= cur_cost_threshold
                # find the true in idx_min_cost_threshold
                idx_min_cost_threshold_true = np.where(idx_min_cost_threshold == True)[0]
                # number of true in idx_min_cost_threshold
                nb_true = idx_min_cost_threshold_true.shape[0]
                # store this value in a list
                nb_true_list.append(nb_true)

            data = dict(
                n_shooting=sub_df["n_shooting"].unique()[0],
                ode_solver_defects=sub_df["ode_solver_defects"].unique()[0],
                number_of_ocp=len(sub_df),
                number_of_near_optimal_ocp=len(sub_df[sub_df["near_optimal"] == True]),
                percent_of_near_optimal_ocp=len(sub_df[sub_df["near_optimal"] == True]) / len(sub_df),
                cumulative_near_optimal_ocp=nb_true_list,
                cumulative_percent_of_near_optimal_ocp=[i / len(sub_df) for i in nb_true_list],
                cumulative_abscissa=[ii * 0.01 for ii in range(0, 200)],
            )
            df_dictionary = pd.DataFrame([data])
            self.near_optimal = pd.concat([self.near_optimal, df_dictionary], ignore_index=True)

    def plot_near_optimality(self, show: bool = True, export: bool = True, export_suffix: str = None):
        """
        This function plots the number of near optimal OCPs for each formulation

        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        """
        # set the n_shooting column as categorical
        df = self.near_optimal.copy()
        # n_shooting as str
        df["n_shooting"] = df["n_shooting"].astype(str)

        fig = px.bar(
            df, x="n_shooting", y="percent_of_near_optimal_ocp", color="ode_solver_defects", barmode="group", height=400
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="black",
        )
        # sh

        # Update axis
        fig.update_xaxes(title_text="Number of nodes")
        fig.update_yaxes(title_text="Percentage of near optimal OCPs (%)")

        # display horinzontal line grid
        fig.update_yaxes(showgrid=True, gridwidth=5)

        # set the colors of the bars with px.colors.qualitative.D3 for each ode_solver
        for i, ode_solver in enumerate(self.near_optimal["ode_solver_defects"].unique()):
            fig.data[i].marker.color = self.colors[ode_solver]

        # bars are transparent a bit
        for i in range(len(fig.data)):
            fig.data[i].marker.opacity = 0.9

        # contours of bars are black
        for i in range(len(fig.data)):
            fig.data[i].marker.line.color = "black"
            fig.data[i].marker.line.width = 1

        # y-axis from 0 to 1
        fig.update_yaxes(range=[0, 1])

        if show:
            fig.show()
        if export:
            self.export(fig, "analyse_near_optimality", export_suffix)

    def plot_near_optimality_cumulative(self, show: bool = True, export: bool = True, export_suffix: str = None):
        """
        This function plots the number of near optimal OCPs for each formulation

        Parameters
        ----------
        show : bool
            If True, the figure is shown.
        export : bool
            If True, the figure is exported.
        """
        # set the n_shooting column as categorical
        df = self.near_optimal.copy()

        fig = go.Figure()
        for ode_solver in self.near_optimal["ode_solver_defects"].unique():
            sub_df = df[df["ode_solver_defects"] == ode_solver]
            fig.add_trace(go.Scatter(
                x=sub_df["cumulative_abscissa"].to_list()[0],
                y=sub_df["cumulative_percent_of_near_optimal_ocp"].to_list()[0],
                mode="lines",
                legendgroup=ode_solver,
                name=ode_solver,
            )
            )
            # fig.show()

        fig.update_layout(
            template="simple_white",
        )
        # sh

        # Update axis
        fig.update_xaxes(title_text="+ x % of the global minimum cost")
        fig.update_yaxes(title_text="Percentage of optimal solutions (%)")

        # display line grid in light grey
        fig.update_yaxes(showgrid=True, gridwidth=2, gridcolor="lightgrey")
        fig.update_xaxes(showgrid=True, gridwidth=2, gridcolor="lightgrey")

        # show ticks on x-axis and y-axis
        fig.update_xaxes(showticklabels=True)
        fig.update_yaxes(showticklabels=True)

        # set the colors of the bars with px.colors.qualitative.D3 for each ode_solver
        # for i, ode_solver in enumerate(self.near_optimal["ode_solver_defects"].unique()):
        #     fig.data[i].marker.color = self.colors[ode_solver]
        #
        # # bars are transparent a bit
        # for i in range(len(fig.data)):
        #     fig.data[i].marker.opacity = 0.9
        #
        # # contours of bars are black
        # for i in range(len(fig.data)):
        #     fig.data[i].marker.line.color = "black"
        #     fig.data[i].marker.line.width = 1
        #
        # # y-axis from 0 to 1
        # fig.update_yaxes(range=[0, 1])

        if show:
            fig.show()
        if export:
            self.export(fig, "analyse_near_optimality_cumulative", export_suffix)

    def animate(self, num: int = 0, export: bool = True):
        """
        This method animates the motion with bioviz

        Parameters
        ----------
        num: int
            Number of the trial to be visualized
        export: bool
            If True, the animation is exported
        """

        print(self.df["filename"].iloc[num])
        print(self.df["grps"].iloc[num])
        path = self.df["model_path"].iloc[num][0] if isinstance(self.df["model_path"].iloc[num], tuple) else self.df["model_path"].iloc[num]

        p = Path(path)
        # verify if the path/file exists with pathlib
        model_path = self.model_path if not p.exists() else p.__str__()
        # remove and change "robot_leg" by "transcriptions"
        model_path = model_path.replace("robot_leg", "transcriptions")

        import bioviz

        s = self.model_path.split("/")[-1].split(".")[0]

        if s == "hexapod_leg":

            b = bioviz.Viz(
                model_path,
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
                mesh_linewidth=5,
            )
            b.resize(1000, 1000)
            b.set_camera_roll(-82.89751054930615)
            b.set_camera_zoom(2.7649491449197656)
            b.set_camera_position(1.266097531449429, -0.6523601622496974, 0.24962580067391163)
            b.set_camera_focus_point(0.07447263939980919, 0.025078204682856153, -0.013568198245759833)

        elif s == "robot_arm":

            b = bioviz.Viz(
                model_path,
                show_now=False,
                show_meshes=True,
                show_global_center_of_mass=False,
                show_gravity_vector=False,
                show_floor=False,
                show_segments_center_of_mass=False,
                show_global_ref_frame=False,
                show_local_ref_frame=False,
                show_markers=True,
                show_muscles=False,
                show_wrappings=False,
                background_color=(1, 1, 1),
                mesh_opacity=0.97,
                mesh_linewidth=5,
            )

            b.resize(1000, 1000)

            b.set_q([-0.15, 0.24, -0.41, 0.21, 0, 0])
            b.set_camera_roll(-84.5816885957667)
            b.set_camera_zoom(2.112003880097381)
            b.set_camera_position(1.9725681105744026, -1.3204979216430117, 0.35790018139336177)
            b.set_camera_focus_point(-0.3283876664932833, 0.5733643134562766, 0.018451815011995998)

        elif s == "acrobat":

            b = bioviz.Viz(
                model_path,
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

            b.set_camera_position(-8.782458942185185, 0.486269131372712, 4.362010279585766)
            b.set_camera_roll(90)
            b.set_camera_zoom(0.308185240948253)
            b.set_camera_focus_point(1.624007185850899, 0.009961251074366406, 1.940316420941989)

            b.resize(600, 900)
        elif s == "wu_converted_definitif_without_floating_base_template_xyz_offset_with_variables":

            b = bioviz.Viz(
                model_path,
                show_now=False,
                show_meshes=True,
                show_global_center_of_mass=False,
                show_gravity_vector=False,
                show_floor=False,
                show_segments_center_of_mass=False,
                show_global_ref_frame=False,
                show_local_ref_frame=False,
                show_markers=False,
                show_muscles=True,
                show_wrappings=False,
                background_color=(1, 1, 1),
                mesh_opacity=0.97,
            )

            b.resize(1000, 1000)

            # b.set_q([-0.15, 0.24, -0.41, 0.21, 0, 0])
            b.set_camera_roll(-100.90843467296737)
            b.set_camera_zoom(1.9919059008044755)
            b.set_camera_position(0.8330547810707182, 2.4792370867179256, 0.1727481994453778)
            b.set_camera_focus_point(-0.2584435804313228, 0.8474543937884143, 0.2124670559215174)

        elif s == "Humanoid10Dof":

            b = bioviz.Viz(
                model_path,
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
                mesh_linewidth=5,
                contacts_size=0.04,
            )

            b.resize(1000, 1000)

            b.set_camera_roll(-91.44517177211645)
            b.set_camera_zoom(0.7961539827851234)
            b.set_camera_position(4.639962934524132, 0.4405891958030146, 0.577705598983718)
            b.set_camera_focus_point(-0.2828701273331326, -0.04065388066757992, 0.9759133347931428)

        # print("roll")
        # print(biorbd_viz.get_camera_roll())
        # print("zoom")
        # print(biorbd_viz.get_camera_zoom())
        # print("position")
        # print(biorbd_viz.get_camera_position())
        # print("get_camera_focus_point")
        # print(biorbd_viz.get_camera_focus_point())
        # Record

        q = self.df["q"].iloc[num]

        if export:
            print(f"{self.path_to_figures}/{s}_video.ogv")
            b.start_recording(f"{self.path_to_figures}/{s}_video.ogv")
            b.load_movement(q)
            for f in range(q.shape[1] + 1):
                b.movement_slider[0].setValue(f)
                b.add_frame()
            b.stop_recording()
            b.quit()
        else:
            b.load_movement(q)
            b.exec()



    def kinogram(self, num: int = 0, nb_frames: int = 5):
        """
        This method animates the motion with bioviz

        Parameters
        ----------
        num: int
        Number of the trial to be visualized
        """
        # Name of the model only the end of the path without extension
        s = self.model_path.split("/")[-1].split(".")[0]

        print(self.df["filename"].iloc[num])
        print(self.df["grps"].iloc[num])

        model_path = self.df["model_path"].iloc[num]
        p = Path(model_path[0] if isinstance(model_path, tuple) else model_path)
        # verify if the path/file exists with pathlib
        model_path = self.model_path if not p.exists() else p.__str__()
        # remove and change "robot_leg" by "transcriptions"
        model_path = model_path.replace("robot_leg", "transcriptions")

        import bioviz

        # Position camera
        if s == "hexapod_leg":

            b = bioviz.Viz(
                model_path,
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
                mesh_linewidth=5,
            )
            b.resize(1000, 1000)
            b.set_camera_roll(-82.89751054930615)
            b.set_camera_zoom(2.7649491449197656)
            b.set_camera_position(1.266097531449429, -0.6523601622496974, 0.24962580067391163)
            b.set_camera_focus_point(0.07447263939980919, 0.025078204682856153, -0.013568198245759833)

        elif s == "robot_arm":

            b = bioviz.Viz(
                model_path,
                show_now=False,
                show_meshes=True,
                show_global_center_of_mass=False,
                show_gravity_vector=False,
                show_floor=False,
                show_segments_center_of_mass=False,
                show_global_ref_frame=False,
                show_local_ref_frame=False,
                show_markers=True,
                show_muscles=False,
                show_wrappings=False,
                background_color=(1, 1, 1),
                mesh_opacity=0.97,
                mesh_linewidth=5,
            )

            b.resize(1000, 1000)

            b.set_q([-0.15, 0.24, -0.41, 0.21, 0, 0])
            b.set_camera_roll(-84.5816885957667)
            b.set_camera_zoom(2.112003880097381)
            b.set_camera_position(1.9725681105744026, -1.3204979216430117, 0.35790018139336177)
            b.set_camera_focus_point(-0.3283876664932833, 0.5733643134562766, 0.018451815011995998)

        elif s == "acrobat":

            b = bioviz.Viz(
                model_path,
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

            b.set_camera_position(-8.782458942185185, 0.486269131372712, 4.362010279585766)
            b.set_camera_roll(90)
            b.set_camera_zoom(0.308185240948253)
            b.set_camera_focus_point(1.624007185850899, 0.009961251074366406, 1.940316420941989)

            b.resize(600, 900)
        elif s == "wu_converted_definitif_without_floating_base_template_xyz_offset_with_variables":

            b = bioviz.Viz(
                model_path,
                show_now=False,
                show_meshes=True,
                show_global_center_of_mass=False,
                show_gravity_vector=False,
                show_floor=False,
                show_segments_center_of_mass=False,
                show_global_ref_frame=False,
                show_local_ref_frame=False,
                show_markers=False,
                show_muscles=True,
                show_wrappings=False,
                background_color=(1, 1, 1),
                mesh_opacity=0.97,
            )

            b.resize(1000, 1000)

            # b.set_q([-0.15, 0.24, -0.41, 0.21, 0, 0])
            b.set_camera_roll(-100.90843467296737)
            b.set_camera_zoom(1.9919059008044755)
            b.set_camera_position(0.8330547810707182, 2.4792370867179256, 0.1727481994453778)
            b.set_camera_focus_point(-0.2584435804313228, 0.8474543937884143, 0.2124670559215174)

        elif s == "Humanoid10Dof":

            b = bioviz.Viz(
                model_path,
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
                mesh_linewidth=5,
                contacts_size=0.04,
            )

            b.resize(1000, 1000)

            b.set_camera_roll(-91.44517177211645)
            b.set_camera_zoom(0.7961539827851234)
            b.set_camera_position(4.639962934524132, 0.4405891958030146, 0.577705598983718)
            b.set_camera_focus_point(-0.2828701273331326, -0.04065388066757992, 0.9759133347931428)

        q = self.df["q"].iloc[num]
        b.load_movement(q)

        print("roll")
        print(b.get_camera_roll())
        print("zoom")
        print(b.get_camera_zoom())
        print("position")
        print(b.get_camera_position())
        print("get_camera_focus_point")
        print(b.get_camera_focus_point())

        start = 0
        end = self.df["n_shooting"].iloc[num]
        step = int((end - start) / nb_frames)

        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        fig, ax = plt.subplots(1,
                               # compute from end and step, the total number of frames
                               len(range(start, end, step)),
                               figsize=(19.20, 5.4))
        fig.subplots_adjust(hspace=0, wspace=0)



        # Taking snapshot
        count = 0
        for snap in range(start, end, step):
            b.movement_slider[0].setValue(snap)
            b.snapshot(f"{self.path_to_figures}/{s}_{snap}.png")
            # b.refresh_window()
            img = mpimg.imread(f"{self.path_to_figures}/{s}_{snap}.png")

            ax[count].xaxis.set_major_locator(plt.NullLocator())
            ax[count].yaxis.set_major_locator(plt.NullLocator())
            ax[count].imshow(img)
            ax[count].spines["top"].set_visible(False)
            ax[count].spines["right"].set_visible(False)
            ax[count].spines["bottom"].set_visible(False)
            ax[count].spines["left"].set_visible(False)
            count += 1

        b.quit()
        fig.show()

        filepath = f"{self.path_to_figures}/kinogram_{s}.svg"
        filepath_parent = f"{self.path_to_figures}/../../kinogram_{s}.svg"
        fig.savefig(filepath, format="svg", dpi=900, bbox_inches="tight")
        fig.savefig(filepath_parent, format="svg", dpi=900, bbox_inches="tight")
        fig.savefig(f"{self.path_to_figures}/kinogram_{s}.png", format="png", dpi=900, bbox_inches="tight")
        plt.close(fig)

        b.exec()

        return filepath

    def plot_time_iter(self, show: bool = True, export: bool = True, time_unit: str = "s", export_suffix: str = None):
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
        dyn = self.df["ode_solver_defects_labels"].unique().tolist()
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
            self.export(fig, "analyse_time_iter", export_suffix)

    def plot_integration_frame_to_frame_error(
            self, show: bool = True, export: bool = True, until_consistent: bool = False, export_suffix: str = None
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
        dyn = self.df["ode_solver_defects_labels"].unique().tolist()
        grps = dyn

        fig = make_subplots(rows=1, cols=2, subplot_titles=["translation error", "rotation error"])
        # update the font size of the subplot_titles
        for i in fig["layout"]["annotations"]:
            i["font"] = dict(size=18)

        # select only the one who converged
        df_results = self.df[self.df["status"] == 0]

        for _, row in df_results.iterrows():
            idx_end = int(row.consistent_threshold) if until_consistent else int(len(row.time))
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
                    color=self.colors[row.ode_solver_defects],
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
                    color=self.colors[row.ode_solver_defects],
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
            self.export(fig, "analyse_integration_each_frame", export_suffix)

    def plot_integration_final_error(self, show: bool = True, export: bool = True, export_suffix: str = None):
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
        dyn = self.df["ode_solver_defects_labels"].unique().tolist()
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
            self.export(fig, "analyse_integration", export_suffix)

    def plot_obj_values(self, show: bool = True, export: bool = True, export_suffix: str = None):
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
        dyn = self.df["ode_solver_defects_labels"].unique().tolist()
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
            self.export(fig, "analyse_obj", export_suffix)

        return fig

    def plot_obj_value_with_consistency(self, threshold: int = 10, show: bool = True, export: bool = True, export_suffix: str = None):
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
        dyn = self.df["ode_solver_defects_labels"].unique().tolist()
        grps = dyn

        fig = self.plot_obj_values(show=False, export=False)

        # select only the one who converged
        df_results = self.df[self.df["status"] == 0]
        # add a stripplot on the figure
        for d in dyn:
            x = df_results["ode_solver_defects_labels"][
                (df_results["ode_solver_defects_labels"] == d)
                & (df_results["rotation_error"] > threshold)
                ]
            y = df_results["cost"][
                (df_results["ode_solver_defects_labels"] == d)
                & (df_results["rotation_error"] > threshold)
                ]
            fig.add_trace(
                go.Box(
                    x=x,
                    y=y,
                    name="else",
                    boxpoints="all",
                    width=0.4,
                    pointpos=-2,
                    legendgroup="else",
                    # fillcolor=c,
                    jitter=0.4,
                    marker=dict(color='black', size=3),
                    line=dict(color='rgba(0,0,0,0)'),
                    fillcolor='rgba(0,0,0,0)',
                ),
                row=1,
                col=1,
            )

        if show:
            fig.show()
        if export:
            self.export(fig, "analyse_obj_consistency", export_suffix)

        return fig

    def plot_obj_jitter_with_consistency(
            self,
            thresholds: List[float] = None,
            show: bool = True,
            export: bool = True,
            export_suffix: str = None
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
        dyn = self.df["ode_solver_defects_labels"].unique().tolist()
        grps = dyn

        fig = make_subplots(rows=1, cols=1)

        # select only the one who converged
        df_results = self.df[self.df["status"] == 0]
        # add a stripplot on the figure
        for i_d, d in enumerate(dyn):
            # init x_tot pd
            x_tot = pd.DataFrame()
            y_tot = pd.DataFrame()
            for i, t in enumerate(thresholds):
                # isolate the data for the given threshold t
                x = df_results["ode_solver_defects_labels"][
                    (df_results["ode_solver_defects_labels"] == d)
                    & (df_results["rotation_error"] > t)
                    ]
                # replace the name in x vector in function of i
                x = x.replace(d, f"{d} - {t}")

                # y
                y = df_results["cost"][
                    (df_results["ode_solver_defects_labels"] == d)
                    & (df_results["rotation_error"] > t)
                    ]
                # concatenate the data for the given threshold t
                x_tot = x if i == 0 else pd.concat([x_tot, x])
                y_tot = y if i == 0 else pd.concat([y_tot, y])

            fig.add_trace(
                go.Box(
                    x=x_tot,
                    y=y_tot,
                    name=d,
                    boxpoints="all",
                    width=0.5,
                    pointpos=0,
                    legendgroup=d,
                    # fillcolor=c,
                    jitter=0.5,
                    marker=dict(color=self.colors[self.ode_solvers[i_d]], size=8, line=dict(width=1,
                                        color='DarkSlateGrey')),
                    line=dict(color='rgba(0,0,0,0)'),
                    fillcolor='rgba(0,0,0,0)',
                ),
                row=1,
                col=1,
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

        if show:
            fig.show()
        if export:
            self.export(fig, "analyse_obj_consistency", export_suffix)

        return fig

    def plot_keys(
            self,
            keys: List[str],
            df_list: List[str] = None,
            ylabel: List[str] = None,
            ylog: List[bool] = None,
            fig: go.Figure = None,
            col: int = 1,
            show: bool = True,
            export: bool = True,
            export_suffix: str = None,
    ):
        if fig is None:
            fig = make_subplots(cols=1, rows=len(keys))

        dyn = self.df["ode_solver_defects_labels"].unique().tolist()
        grps = dyn

        # select only the one who converged
        df_results = self.df[self.df["status"] == 0]

        for i, key in enumerate(keys):
            row = i + 1
            if df_list is None or df_list[i] == "df":
                fig = my_traces(
                    fig,
                    dyn,
                    grps,
                    df_results,
                    key=key,
                    row=row,
                    col=col,
                    ylabel=ylabel[i] if ylabel is not None else ylabel,
                    ylog=ylog[i] if ylog is not None else True,
                    colors=[self.colors[ode] for ode in self.ode_solvers],
                )
            elif df_list[i] == "near_optimal":
                df = self.near_optimal.copy()

                if key == "cumulative_percent_of_near_optimal_ocp":

                    for j, ode in enumerate(self.ode_solvers):
                        sub_df = df[df["ode_solver_defects"] == ode]
                        if len(sub_df) == 0:
                            continue

                        fig.add_trace(go.Scatter(
                            x=sub_df["cumulative_abscissa"].to_list()[0],
                            y=sub_df["cumulative_percent_of_near_optimal_ocp"].to_list()[0],
                            mode="lines",
                            legendgroup=grps[j],
                            name=grps[j],
                            showlegend=show,
                        ),
                            row=row,
                            col=col,
                        )

                        line_colors = self.colors[ode]
                        fig.update_traces(
                            line=dict(color=line_colors),
                            selector=dict(legendgroup=grps[j]),
                        )

                    fig.update_layout(
                        template="simple_white",
                    )
                    # sh

                    # Update axis
                    fig.update_xaxes(title_text="+ x % of the global minimum cost", row=row, col=col)
                    fig.update_yaxes(title_text="Optimal solutions (%)", row=row, col=col)

                    # display line grid in light grey
                    fig.update_yaxes(showgrid=True, gridwidth=2, gridcolor="lightgrey", row=row, col=col)
                    fig.update_xaxes(showgrid=True, gridwidth=2, gridcolor="lightgrey", row=row, col=col)

                    # show ticks on x-axis and y-axis
                    fig.update_xaxes(showticklabels=True, row=row, col=col)
                    fig.update_yaxes(showticklabels=True, row=row, col=col)

                else:
                    # n_shooting as str
                    df["n_shooting"] = df["n_shooting"].astype(str)

                    for j, ode in enumerate(self.ode_solvers):
                        df_ode = df[df["ode_solver_defects"] == ode]
                        fig = fig.add_trace(
                            go.Bar(
                                x=[1],
                                y=df_ode[key],
                                legendgroup=grps[j],
                                showlegend=False,
                            ),
                            row=row,
                            col=col,
                        )
                        marker_colors = self.colors[ode]
                        # udpate the color of the bar
                        fig.data[-1].marker.color = marker_colors
                        # opacity
                        fig.data[-1].opacity = 0.75

                    # hide x ticks
                    fig.update_xaxes(showticklabels=False, row=row, col=col)

                    # Update axis
                    fig.update_yaxes(title_text=ylabel[i] if ylabel is not None else ylabel, row=row, col=col)

                    # y-axis from 0 to 1
                    fig.update_yaxes(range=[0, 1], row=row, col=col)

        return fig

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
            export_suffix: str = None,
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
        dyn = self.df["ode_solver_defects_labels"].unique().tolist()
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
            colors=[self.colors[ode] for ode in self.ode_solvers],
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

    def plot_cost_vs_consistency(self, show: bool = True, export: bool = True, export_suffix: str = None):
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

        fig.update_layout(
            height=700,
            width=800,
        )
        fig.update_layout(
            font=dict(
                size=20,
                family="Times New Roman",
            ),
            legend=dict(
                font=dict(
                    size=20,
                    family="Times New Roman",
                ),
                yanchor="bottom",
                y=0.01,
                xanchor="center",
                x=0.5,
            ),
        )

        if show:
            fig.show()
        if export:
            self.export(fig, "analyse_time_vs_obj", export_suffix)

    def plot_time_vs_consistency(self, show: bool = True, export: bool = True, export_suffix: str = None):
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
            self.export(fig, "analyse_time_vs_consistency", export_suffix)

    def export(self, fig: go.Figure, filename: str, export_suffix: str = None):
        """
        This function export the results in a csv file

        Parameters
        ----------
        fig : go.Figure
            The figure to export
        filename : str
            The name of the file
        export_suffix : str
            The suffix to add to the file name
        """

        format_type = ["png", "pdf", "svg", "eps"]
        for f in format_type:
            fig.write_image(self.path_to_figures + f"/{filename}{export_suffix}." + f)
        fig.write_html(self.path_to_figures + f"/{filename}{export_suffix}.html", include_mathjax="cdn")

    def plot_time_vs_obj(self, show: bool = True, export: bool = True, export_suffix: str = None):
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
            fig.write_html(self.path_to_figures + f"/analyse_time_vs_obj{export_suffix}.html", include_mathjax="cdn")

    def plot_detailed_obj_values(self, show: bool = True, export: bool = True, export_suffix: str = None):
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

        # select only the one who converged
        df_results = self.df[self.df["status"] == 0]

        nb_costs = df_results["detailed_cost"].iloc[0].__len__()
        rows, cols = generate_windows_size(nb_costs)

        # indices of rows and cols for each axe of the subplot
        idx_rows, idx_cols = np.unravel_index([i for i in range(nb_costs)], (rows, cols))
        idx_rows += 1
        idx_cols += 1

        titles = []
        for i in range(nb_costs):
            name = df_results[f"cost{i}_details"].iloc[0]["name"]
            # key = " " if df_results[f"cost{i}_details"][0]["params"]["key"] is not None else ""
            # if param is not empty, key is ""
            param = df_results[f"cost{i}_details"].iloc[0]["params"]
            key = " " + param["key"] if param and "key" in param.keys() else ""

            derivative = "delta " if df_results[f"cost{i}_details"].iloc[0]["derivative"] else ""
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
                # title_str=df_results[f"cost{i}name"][0] + " " + key,
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
            fig.write_html(self.path_to_figures + f"/analyse_obj{export_suffix}.html", include_mathjax="cdn")

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
            export_suffix: str = None,
    ) -> go.Figure:
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
        fig :go.Figure
            Figure object.
        """
        model = biorbd.Model(self.model_path)

        if "tau" in key:
            nq = 9
            list_dof = [dof.to_string() for dof in model.nameDof()][6:]
        else:
            nq = model.nbQ()
            list_dof = [dof.to_string() for dof in model.nameDof()]

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
        trans_idx, rot_idx = get_trans_and_rot_idx(model)

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
        trans_idx, rot_idx = get_trans_and_rot_idx(model)

        if "tau" in key:
            trans_idx = []
            rot_idx = rot_idx[:-6]

        for idx in trans_idx:
            fig.update_yaxes(row=idx_rows[idx], col=idx_cols[idx], title=ylabel_translations)
        for idx in rot_idx:
            fig.update_yaxes(row=idx_rows[idx], col=idx_cols[idx], title=ylabel_rotations)

        if show:
            fig.show()
        if export:
            format_type = ["png", "pdf", "svg", "eps"]
            for f in format_type:
                fig.write_image(self.path_to_figures + f"/analyse_{key}{export_suffix}." + f)
            fig.write_html(self.path_to_figures + f"/analyse_{key}{export_suffix}.html", include_mathjax="cdn")

        return fig

    def analyse(self, show=True, export=True, cluster_analyse=False):
        self.print()
        self.plot_near_optimality(show=show, export=export)
        # self.plot_time_iter(show=show, export=export, time_unit="min")
        self.plot_obj_values(show=show, export=export)
        self.plot_detailed_obj_values(show=show, export=export)
        # self.plot_time_vs_obj(show=show, export=export)
        # self.plot_time_vs_consistency(show=show, export=export)
        # self.plot_cost_vs_consistency(show=show, export=export)
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
                path_to_files=self.path_to_files, model_path=self.model_path, df=self.df[self.df["cluster"] == i]
            )
            if animate:
                cluster_results.animate()
            # cluster_results.plot_time_iter(show=show, export=export, time_unit="min", export_suffix=export_suffix)
            cluster_results.plot_obj_values(show=show, export=export, export_suffix=export_suffix)
            # cluster_results.plot_detailed_obj_values(show=show, export=export, export_suffix=export_suffix)
            cluster_results.plot_near_optimality(show=show, export=export, export_suffix=export_suffix)

            # cluster_results.plot_time_vs_obj(show=show, export=export, export_suffix=export_suffix)
            # cluster_results.plot_time_vs_consistency(show=show, export=export, export_suffix=export_suffix)
            # cluster_results.plot_cost_vs_consistency(show=show, export=export, export_suffix=export_suffix)

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


def generate_results_objects():
    results_leg = ResultsAnalyse.from_folder(
        model_path=Models.LEG.value,
        path_to_files=ResultFolders.LEG_100.value,
        export=True,
    )
    results_leg.print()
    # export the entire object results_leg in a pickle file
    with open("results_leg.pickle", "wb") as f:
        pickle.dump(results_leg, f)

    results_arm = ResultsAnalyse.from_folder(
        model_path=Models.ARM.value,
        path_to_files=ResultFolders.ARM_100.value,
        export=True,
    )
    results_arm.print()
    # export the entire object results_arm in a pickle file
    with open("results_arm.pickle", "wb") as f:
        pickle.dump(results_arm, f)

    results_acrobat = ResultsAnalyse.from_folder(
        model_path=Models.ACROBAT.value,
        path_to_files=ResultFolders.ACROBAT_100.value,
        export=True,
    )
    results_acrobat.print()
    # export the entire object results_acrobat in a pickle file
    with open("results_acrobat.pickle", "wb") as f:
        pickle.dump(results_acrobat, f)

    results_walker = ResultsAnalyse.from_folder(
        model_path=Models.HUMANOID_10DOF.value,
        path_to_files=ResultFolders.WALKING_100.value,
        export=True,
    )
    results_walker.print()
    # export the entire object results_walker in a pickle file
    with open("results_walker.pickle", "wb") as f:
        pickle.dump(results_walker, f)

    results_upper_limb = ResultsAnalyse.from_folder(
        model_path=Models.UPPER_LIMB_XYZ_VARIABLES.value,
        path_to_files=ResultFolders.UPPER_LIMB_100.value,
        export=True,
    )
    results_upper_limb.print()
    # export the entire object results_walker in a pickle file
    with open("results_upper_limb.pickle", "wb") as f:
        pickle.dump(results_upper_limb, f)

    return results_leg, results_arm, results_acrobat, results_walker, results_upper_limb


def load_results_objects():
    with open("results_leg.pickle", "rb") as f:
        results_leg = pickle.load(f)
        results_leg.print()
    with open("results_arm.pickle", "rb") as f:
        results_arm = pickle.load(f)
        results_arm.print()
    with open("results_acrobat.pickle", "rb") as f:
        results_acrobat = pickle.load(f)
        results_acrobat.print()
    with open("results_walker.pickle", "rb") as f:
        results_walker = pickle.load(f)
        results_walker.print()
    with open("results_upper_limb.pickle", "rb") as f:
        results_upper_limb = pickle.load(f)
        results_upper_limb.print()

    return results_leg, results_arm, results_acrobat, results_walker, results_upper_limb


def big_figure(
        results_leg: ResultsAnalyse,
        results_arm: ResultsAnalyse,
        results_acrobat: ResultsAnalyse,
        results_walker: ResultsAnalyse,
        results_upper_limb: ResultsAnalyse,
):
    fig = make_subplots(
        rows=4,
        cols=5,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        subplot_titles=(
            "OCP1 - Leg",
            "OCP2 - Arm",
            "OCP3 - Upper limb",
            "OCP4 - Planar human",
            "OCP5 - Acrobat"),
    )

    df = ["df", "df", "near_optimal", "df"]
    keys = ["computation_time", "cost", "cumulative_percent_of_near_optimal_ocp", "rotation_error"]
    ylabels = ["CPU time\n(s)", "Cost function value", "Near optimal frequency (%)", "Rotation error RMSE\n(deg)"]

    # df = ["df", "df", "df"]
    # keys = ["computation_time", "cost", "rotation_error"]
    # ylabels = ["CPU time\n(s)", "Cost function value", "Rotation error RMSE\n(deg)"]

    ylog = [False, True, False, True]
    fig = results_leg.plot_keys(keys=keys, fig=fig, col=1, ylabel=ylabels, df_list=df, ylog=ylog)
    fig = results_arm.plot_keys(keys=keys, fig=fig, col=2, ylog=ylog, df_list=df)
    fig = results_upper_limb.plot_keys(keys=keys, fig=fig, col=3, ylog=ylog, df_list=df)
    fig = results_walker.plot_keys(keys=keys, fig=fig, col=4, ylog=ylog, df_list=df)
    fig = results_acrobat.plot_keys(keys=keys, fig=fig, col=5, ylog=ylog, df_list=df)

    fig.update_layout(
        height=900,
        width=1200,
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

    # display the horizontal lines for each grid of the figure
    for i in range(1, 5):
        for j in range(1, 5):
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgrey", row=i, col=j)

    # delete all the yaxis titles if col > 1
    for i in range(1, 5):
        for j in range(2, 6):
            fig.update_yaxes(title="", row=i, col=j)

    # custom ranges
    fig.update_yaxes(range=[0, 4], row=1, col=1)
    fig.update_yaxes(range=[np.log10(7.21e-4), np.log10(7.214e-4)], row=2, col=1)
    fig.update_yaxes(range=[np.log10(884), np.log10(884.2)], row=2, col=4)
    fig.update_yaxes(range=[0, 0.6e4], row=1, col=3)

    # same format for all y ticks
    for i in range(6):
        fig.update_yaxes(tickformat=".2e", row=2, col=i)

    for i in range(6):
        fig.update_yaxes(tickformat=".0%", row=5, col=i)
        fig.update_xaxes(tickformat=".0%", row=5, col=i)

    for i in range(6):
        fig.update_xaxes(range=[0, 1.6], row=5, col=i)

    # legend font bigger
    fig.update_layout(legend=dict(font=dict(size=15)))

    fig.show()


def figure_article(
        results_leg: ResultsAnalyse,
        results_arm: ResultsAnalyse,
        results_acrobat: ResultsAnalyse,
        results_walker: ResultsAnalyse,
        results_upper_limb: ResultsAnalyse,
):
    fig = make_subplots(
        rows=3,
        cols=5,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        subplot_titles=(
            "<b>OCP1 - Leg</b>",
            "<b>OCP2 - Arm</b>",
            "<b>OCP3 - Upper limb</b>",
            "<b>OCP4 - Planar human</b>",
            "<b>OCP5 - Acrobat</b>"),
    )

    df = ["df", "df", "df"]
    keys = ["computation_time", "cost", "rotation_error"]
    ylabels = ["CPU time\n(s)", "   Cost function value", "Rotation error RMSE\n(deg)"]

    ylog = [False, True, True]
    fig = results_leg.plot_keys(keys=keys, fig=fig, col=1, ylabel=ylabels, df_list=df, ylog=ylog)
    # move the ylabel to the left to avoid overlapping with yticks

    fig = results_arm.plot_keys(keys=keys, fig=fig, col=2, ylog=ylog, df_list=df)
    fig = results_walker.plot_keys(keys=keys, fig=fig, col=4, ylog=ylog, df_list=df)
    fig = results_upper_limb.plot_keys(keys=keys, fig=fig, col=3, ylog=ylog, df_list=df)
    fig = results_acrobat.plot_keys(keys=keys, fig=fig, col=5, ylog=ylog, df_list=df)

    fig.update_layout(
        height=600,
        width=1000,
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

    # display the horizontal lines for each grid of the figure
    for i in range(1, 6):
        for j in range(1, 6):
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgrey", row=i, col=j)

    # 5 ticks on y axis for each subplots
    for i in range(1, 6):
        for j in range(1, 6):
            fig.update_yaxes(nticks=5, row=i, col=j)

    # delete all the yaxis titles if col > 1
    for i in range(1, 5):
        for j in range(2, 6):
            fig.update_yaxes(title="", row=i, col=j)

    # all figure from row 1 start from 0 in ordinal axis and the max limit is automatic
    for i in range(1, 6):
        fig.update_yaxes(range=[0, None], row=1, col=i)
        # show the tick 0 in the y axis
        fig.update_yaxes(tick0=0, row=1, col=i)

    # custom ranges
    fig.update_yaxes(range=[0, 4], row=1, col=1)
    fig.update_yaxes(range=[-1, 1000], nticks=4, tick0=0, row=1, col=2)
    fig.update_yaxes(range=[-1, 350], row=1, col=4)
    fig.update_yaxes(range=[-1, 1600], row=1, col=3)

    fig.update_yaxes(range=[np.log10(7.21e-4), np.log10(7.22e-4)], row=2, col=1)
    fig.update_yaxes(range=[np.log10(884), np.log10(884.2)], row=2, col=4)
    fig.update_yaxes(range=[0, 0.6e4], row=1, col=5)

    fig.update_yaxes(range=[-9, -5], row=3, col=4)
    fig.update_yaxes(range=[np.log10(0.005), np.log10(0.01)], row=3, col=3)


    fig.update_yaxes(title_standoff=40, row=1, col=1)
    fig.update_yaxes(title_standoff=0, row=2, col=1)
    fig.update_yaxes(title_standoff=24, row=5, col=1)
    # legend font bigger
    fig.update_layout(legend=dict(font=dict(size=13)))

    # add annotation for the figure 7.24535e+2 on row=2, col=5,
    fig.add_annotation(
        x=0,
        y=1.08,
        xref="x domain",
        yref="y domain",
        text="+7.245352e+2",
        showarrow=False,
        font=dict(size=12),
        xanchor="center",
        yanchor="middle",
        row=2,
        col=3,
    )

    fig.update_yaxes(showexponent="none", row=2, col=3)
    # replace the ticklabels of the y axis of row=2, col=5
    fig.update_yaxes(
        ticktext=["5.4e-7", "5.2e-7", "5.0e-7", "4.8e-7", "4.6e-7", "4.4e-7"],
        tickvals=[7.24535254e+2, 7.24535252e+2, 7.2453525e+2, 7.24535248e+2, 7.24535246e+2, 7.24535244e+2],
        row=2,
        col=3,
    )

    # add annotation for the figure 7.24535e+2 on row=2, col=1,
    fig.add_annotation(
        x=0,
        y=1.08,
        xref="x domain",
        yref="y domain",
        text="+7.21e-4",
        showarrow=False,
        font=dict(size=12),
        xanchor="center",
        yanchor="middle",
        row=2,
        col=1,
    )
    #
    fig.update_yaxes(showexponent="none", row=2,col=1)
    # replace the ticklabels of the y axis of row=2, col=1
    fig.update_yaxes(
        ticktext=["0e-7", "2e-7", "4e-7", "6e-7", "8e-7", "1e-6"],
        tickvals=[7.210e-4, 7.212e-4, 7.214e-4, 7.216e-4, 7.218e-4, 7.22e-4],
        row=2,
        col=1,
    )

    #
    fig.update_yaxes(row=1, col=1, title_standoff=13, anchor="x")
    fig.update_yaxes(row=2, col=1, title_standoff=0, anchor="x")
    fig.update_yaxes(row=3, col=1, title_standoff=0, anchor="x")
    fig.show()

    # export the figure
    filename = "summary_figure"
    export_suffix = ""
    format_type = ["png", "pdf", "svg", "eps"]
    print(f"exported in {Path(results_leg.path_to_files).parent.__str__()}")
    for f in format_type:
        fig.write_image(Path(results_leg.path_to_files).parent.__str__() + f"/{filename}{export_suffix}." + f)
    fig.write_html(Path(results_leg.path_to_files).parent.__str__() + f"/{filename}{export_suffix}.html", include_mathjax="cdn")

def figure_article_split1(
        results_leg: ResultsAnalyse,
        results_arm: ResultsAnalyse,
):
    fig = make_subplots(
        rows=3,
        cols=2,
        vertical_spacing=0.1,
        horizontal_spacing=0.15,
        subplot_titles=(
            "<b>OCP1 - Leg</b>",
            "<b>OCP2 - Arm</b>")
    )

    df = ["df", "df", "df"]
    keys = ["computation_time", "cost", "rotation_error"]
    ylabels = ["CPU time\n(s)", "   Cost function value", "Rotation error RMSE\n(deg)"]

    ylog = [False, True, True]
    fig = results_leg.plot_keys(keys=keys, fig=fig, col=1, ylabel=ylabels, df_list=df, ylog=ylog)
    # move the ylabel to the left to avoid overlapping with yticks
    fig = results_arm.plot_keys(keys=keys, fig=fig, col=2, ylog=ylog, df_list=df, ylabel=["" for i in range(3)])

    fig.update_layout(
        height=600,
        width=600,
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

    # display the horizontal lines for each grid of the figure
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgrey", row=i, col=j)

    # 5 ticks on y axis for each subplots
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(nticks=5, row=i, col=j)

    # delete all the yaxis titles if col > 1
    for i in range(1, 2):
        for j in range(2, 3):
            fig.update_yaxes(title="", row=i, col=j)

    # all figure from row 1 start from 0 in ordinal axis and the max limit is automatic
    for i in range(1, 3):
        fig.update_yaxes(range=[0, None], row=1, col=i)
        # show the tick 0 in the y axis
        fig.update_yaxes(tick0=0, row=1, col=i)

    # custom ranges
    fig.update_yaxes(range=[0, 4], row=1, col=1)
    fig.update_yaxes(range=[-1, 1000], nticks=4, tick0=0, row=1, col=2)

    fig.update_yaxes(range=[np.log10(7.21e-4), np.log10(7.22e-4)], row=2, col=1)


    fig.update_yaxes(title_standoff=40, row=1, col=1)
    fig.update_yaxes(title_standoff=0, row=2, col=1)
    fig.update_yaxes(title_standoff=24, row=5, col=1)
    # legend font bigger
    fig.update_layout(legend=dict(font=dict(size=13)))

    # add annotation for the figure 7.24535e+2 on row=2, col=1,
    fig.add_annotation(
        x=0,
        y=1.08,
        xref="x domain",
        yref="y domain",
        text="+7.21e-4",
        showarrow=False,
        font=dict(size=12),
        xanchor="center",
        yanchor="middle",
        row=2,
        col=1,
    )
    #
    fig.update_yaxes(showexponent="none", row=2,col=1)
    # replace the ticklabels of the y axis of row=2, col=1
    fig.update_yaxes(
        ticktext=["0e-7", "2e-7", "4e-7", "6e-7", "8e-7", "1e-6"],
        tickvals=[7.210e-4, 7.212e-4, 7.214e-4, 7.216e-4, 7.218e-4, 7.22e-4],
        row=2,
        col=1,
    )

    #
    fig.update_yaxes(row=1, col=1, title_standoff=13, anchor="x")
    fig.update_yaxes(row=2, col=1, title_standoff=0, anchor="x")
    fig.update_yaxes(row=3, col=1, title_standoff=0, anchor="x")
    fig.show()

    # export the figure
    filename = "summary_figure"
    export_suffix = ""
    format_type = ["png", "pdf", "svg", "eps"]
    print(f"exported in {Path(results_leg.path_to_files).parent.__str__()}")
    for f in format_type:
        fig.write_image(Path(results_leg.path_to_files).parent.__str__() + f"/{filename}_split1{export_suffix}." + f)
    fig.write_html(Path(results_leg.path_to_files).parent.__str__() + f"/{filename}_split1{export_suffix}.html", include_mathjax="cdn")

def figure_article_split2(
        results_acrobat: ResultsAnalyse,
        results_walker: ResultsAnalyse,
        results_upper_limb: ResultsAnalyse,
):
    fig = make_subplots(
        rows=3,
        cols=3,
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        subplot_titles=(
            "<b>OCP3 - Upper limb</b>",
            "<b>OCP4 - Planar human</b>",
            "<b>OCP5 - Acrobat</b>"),
    )

    df = ["df", "df", "df"]
    keys = ["computation_time", "cost", "rotation_error"]
    ylabels = ["CPU time\n(s)", "   Cost function value", "Rotation error RMSE\n(deg)"]

    ylog = [False, True, True]
    fig = results_upper_limb.plot_keys(keys=keys, fig=fig, col=3 - 2, ylog=ylog, df_list=df, ylabel=ylabels)
    fig = results_walker.plot_keys(keys=keys, fig=fig, col=4-2, ylog=ylog, df_list=df)
    fig = results_acrobat.plot_keys(keys=keys, fig=fig, col=5-2, ylog=ylog, df_list=df)

    fig.update_layout(
        height=600,
        width=700,
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

    # display the horizontal lines for each grid of the figure
    for i in range(1, 4):
        for j in range(3, 6):
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgrey", row=i, col=j-2)

    # 5 ticks on y axis for each subplots
    for i in range(1, 4):
        for j in range(3, 6):
            fig.update_yaxes(nticks=5, row=i, col=j-2)

    # delete all the yaxis titles if col > 1
    for i in range(1, 4):
        for j in range(4, 6):
            fig.update_yaxes(title="", row=i, col=j-2)

    # all figure from row 1 start from 0 in ordinal axis and the max limit is automatic
    for i in range(3, 6):
        fig.update_yaxes(range=[0, None], row=1, col=i-2)
        # show the tick 0 in the y axis
        fig.update_yaxes(tick0=0, row=1, col=i-2)

    # custom ranges
    fig.update_yaxes(range=[-1, 350], row=1, col=4-2)
    fig.update_yaxes(range=[-1, 1600], row=1, col=3-2)

    fig.update_yaxes(range=[np.log10(884), np.log10(884.2)], row=2, col=4-2)
    fig.update_yaxes(range=[0, 0.6e4], row=1, col=5-2)

    fig.update_yaxes(range=[-9, -5], row=3, col=4-2)
    fig.update_yaxes(range=[np.log10(0.005), np.log10(0.01)], row=3, col=3-2)

    # legend font bigger
    fig.update_layout(legend=dict(font=dict(size=13)))

    # add annotation for the figure 7.24535e+2 on row=2, col=5,
    fig.add_annotation(
        x=0,
        y=1.08,
        xref="x domain",
        yref="y domain",
        text="+7.245352e+2",
        showarrow=False,
        font=dict(size=12),
        xanchor="center",
        yanchor="middle",
        row=2,
        col=3-2,
    )

    fig.update_yaxes(showexponent="none", row=2, col=3-2)
    # replace the ticklabels of the y axis of row=2, col=5
    fig.update_yaxes(
        ticktext=["5.4e-7", "5.2e-7", "5.0e-7", "4.8e-7", "4.6e-7", "4.4e-7"],
        tickvals=[7.24535254e+2, 7.24535252e+2, 7.2453525e+2, 7.24535248e+2, 7.24535246e+2, 7.24535244e+2],
        row=2,
        col=3-2,
    )

    fig.show()

    # export the figure
    filename = "summary_figure"
    export_suffix = ""
    format_type = ["png", "pdf", "svg", "eps"]
    print(f"exported in {Path(results_leg.path_to_files).parent.__str__()}")
    for f in format_type:
        fig.write_image(Path(results_leg.path_to_files).parent.__str__() + f"/{filename}_split2{export_suffix}." + f)
    fig.write_html(Path(results_leg.path_to_files).parent.__str__() + f"/{filename}_split2{export_suffix}.html", include_mathjax="cdn")


def figure_cumulative(
        results_arm: ResultsAnalyse,
        results_acrobat: ResultsAnalyse,
):
    fig = make_subplots(
        rows=1,
        cols=2,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        subplot_titles=("6-DoF Arm", "Acrobat"),
    )

    df = ["near_optimal"]
    keys = ["cumulative_percent_of_near_optimal_ocp"]
    ylabels = ["Near optimal frequency (%)"]

    ylog = [False]
    fig = results_arm.plot_keys(keys=keys, fig=fig, col=1, ylog=ylog, df_list=df, show=True)
    fig = results_acrobat.plot_keys(keys=keys, fig=fig, col=2, ylog=ylog, df_list=df, show=False)

    fig.update_layout(
        height=400,
        width=850,
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        legend=dict(
            title_font_family="Times New Roman",
            font=dict(family="Times New Roman", color="black", size=12),
            orientation="h",
            xanchor="center",
            x=0.5,
            y=-0.3,
        ),
        font=dict(
            size=12,
            family="Times New Roman",
        ),
        yaxis=dict(color="black"),
        template="simple_white",
        boxgap=0.2,
    )

    # display the horizontal lines for each grid of the figure
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgrey", row=i, col=j)

    # 5 ticks on y axis for each subplots
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(nticks=5, row=i, col=j)

    # delete all the yaxis titles if col > 1
    for i in range(1, 5):
        for j in range(2, 6):
            fig.update_yaxes(title="", row=i, col=j)

    for i in range(1, 3):
        fig.update_yaxes(tickformat=".0%", row=1, col=i)
        fig.update_yaxes(range=[0, 1.1], row=1, col=i)

    for i in range(1, 3):
        fig.update_xaxes(range=[0, 1.6], row=1, col=i)


    # custom ranges

    fig.show()

    # export the figure
    filename = "cumlative_global_optimum"
    export_suffix = ""
    format_type = ["png", "pdf", "svg", "eps"]
    for f in format_type:
        fig.write_image(Path(results_arm.path_to_files).parent.__str__() + f"/{filename}{export_suffix}." + f, engine="kaleido")
    fig.write_html(Path(results_arm.path_to_files).parent.__str__() + f"/{filename}{export_suffix}.html",
                   include_mathjax="cdn")


if __name__ == "__main__":
    # results_leg, results_arm, results_acrobat, results_walker, results_upper_limb = generate_results_objects()

    # results_upper_limb = ResultsAnalyse.from_folder(
    #     model_path=Models.UPPER_LIMB_XYZ_VARIABLES.value,
    #     path_to_files=ResultFolders.UPPER_LIMB_100.value,
    #     export=True,
    # )
    # results_upper_limb.print()
    # # export the entire object results_walker in a pickle file
    # with open("results_upper_limb.pickle", "wb") as f:
    #     pickle.dump(results_upper_limb, f)

    results_leg, results_arm, results_acrobat, results_walker, results_upper_limb = load_results_objects()

    # results_leg.plot_near_optimality_cumulative(show=True, export=True)
    # results_arm.plot_near_optimality_cumulative(show=True, export=True)
    # results_acrobat.plot_near_optimality_cumulative(show=True, export=True)

    # results_acrobat.plot_obj_value_with_consistency(threshold=50)
    # results_acrobat.plot_obj_jitter_with_consistency(thresholds=[0, 50])
    #

    # figure_article(
    #     results_leg=results_leg,
    #     results_arm=results_arm,
    #     results_acrobat=results_acrobat,
    #     results_walker=results_walker,
    #     results_upper_limb=results_upper_limb,
    # )

    figure_article_split1(
        results_leg=results_leg,
        results_arm=results_arm,
    )
    figure_article_split2(
        results_acrobat=results_acrobat,
        results_walker=results_walker,
        results_upper_limb=results_upper_limb,
    )
    # results_leg.plot_obj_values(show=True, export=True)
    # results_leg.plot_state(key="tau", show=True, export=True)
    # results_leg.plot_state(key="q", show=True, export=True)
    # results_leg.plot_state(key="qdot", show=True, export=True)
    #
    # # results_arm.plot_state(key="tau", show=True, export=True)
    # # results_arm.plot_state(key="q", show=True, export=True)
    # # results_arm.plot_state(key="qdot", show=True, export=True)
    #
    # results_walker.plot_obj_values(show=True, export=True)
    # results_walker.plot_state(key="tau", show=True, export=True)
    # results_walker.plot_state(key="q", show=True, export=True)
    # results_walker.plot_state(key="qdot", show=True, export=True)
    #
    # results_upper_limb.plot_state(key="tau", show=True, export=True)
    # results_upper_limb.plot_state(key="q", show=True, export=True)
    # results_upper_limb.plot_state(key="qdot", show=True, export=True)

    # results_acrobat.plot_cost_vs_consistency(show=True, export=True)

    # figure_cumulative(
    #     results_arm=results_arm,
    #     results_acrobat=results_acrobat,
    # )
    image_paths = []
    # image_paths += results_leg.kinogram(num=0, nb_frames=5)
    # image_paths += results_arm.kinogram(num=0, nb_frames=5)
    # image_paths += results_acrobat.kinogram(num=0, nb_frames=5)
    image_paths += results_acrobat.kinogram(num=0, nb_frames=9)
    # image_paths += results_upper_limb.kinogram(num=0, nb_frames=5)
    # image_paths += results_walker.kinogram(num=100, nb_frames=5)


    # results_acrobat.plot_cost_vs_consistency(show=True, export=True)

    # results_leg.analyse(
    #     show=True,
    #     export=True,
    # )

    # results_leg.animate(num=5, export=True)
    # results_arm.animate(num=5, export=True)
    # results_acrobat.animate(num=5, export=True)
    # results_walker.animate(num=5, export=True)
    # results_upper_limb.animate(num=5, export=True)
    # results_arm.analyse(
    #     show=True,
    #     export=True,
    #     cluster_analyse=True,
    # )
    # results_arm.cluster_analyse(
    #     show=True,
    # )

    # results_acrobat.analyse(
    #     show=True,
    #     export=True,
    # )
