"""
This script is reading and organizing the raw data results from Miller Optimal control problems into a nice DataFrame.
It requires the all the raw data to run the script.
"""
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import plotly
from plotly.subplots import make_subplots
import plotly.express as px

import biorbd

from utils import (
    stack_states,
    stack_controls,
    define_time,
    compute_error_single_shooting,
    compute_error_single_shooting_each_frame,
    my_traces,
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
        export: bool = True,
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
        self.df_path = None
        self.df = self.to_panda_dataframe(
            export=export,
        )

    def to_panda_dataframe(self, export: bool = True) -> pd.DataFrame:
        """
        Convert the data to a pandas dataframe.

        Parameters
        ----------
        export : bool
            Export the dataframe as a pickle file.

        Returns
        -------
        df_results : pd.DataFrame
            The pandas dataframe with the results.
        """

        # open files
        files = os.listdir(self.path_to_files)
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
                p = Path(f"{self.path_to_files}/{file}")
                file_path = open(p, "rb")
                data = pickle.load(file_path)

                # DM to array
                data["cost"] = np.array(data["cost"])[0][0]
                # print(data["n_threads"])
                # compute error
                model = biorbd.Model(self.model_path)

                # if isinstance(data["n_shooting"], tuple) and len(data["n_shooting"]) > 1:
                #     n_shooting = sum(data["n_shooting"])
                #     q = stack_states(data["q"], "q")  # todo: "q" shouldn't be there.
                #     q_integrated = data["q_integrated"]["q"]  # todo: "q" shouldn't be there.
                #     data["q"] = q
                #     data["q_integrated"] = q_integrated
                # # # else:
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
                data["consistent_threshold"] = np.where(
                    data["rotation_error_traj"] > self.consistent_threshold
                )[0][0]

                data[
                    "grps"
                ] = f"{data['ode_solver'].__str__()}_{data['defects_type'].value}_{n_shooting}"

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

        # saves the dataframe
        if export:
            self.df_path = f"{self.path_to_data}/Dataframe_results_metrics.pkl"
            df_results.to_pickle(self.df_path)

        return df_results

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
            str_formulation = f.replace("_", " ").replace("-", " ").replace("\n", " ")
            a = len(self.df[(self.df["status"] == 1) & (self.df["grps"] == f)])
            b = len(self.df[self.df["grps"] == f])
            print(
                f"{a} / {b} {str_formulation} did not converge to an optimal solutions"
            )

    def plot_time_iter(
        self, show: bool = True, export: bool = True, time_unit: str = "s"
    ):
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
            r"$\text{time ({time_unit}})}$",
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
            tickformat="0.1r",
        )
        fig.update_yaxes(
            row=1,
            col=2,
            tickformat=".0f",
        )

        if show:
            fig.show()
        if export:
            fig.write_image(self.path_to_figures + "/analyse_time_iter.png")
            fig.write_image(self.path_to_figures + "/analyse_time_iter.pdf")
            fig.write_image(self.path_to_figures + "/analyse_time_iter.svg")
            fig.write_image(self.path_to_figures + "/analyse_time_iter.eps")
            fig.write_html(
                self.path_to_figures + "/analyse_time_iter.html", include_mathjax="cdn"
            )

    def plot_integration_frame_to_frame_error(
        self, show: bool = True, export: bool = True, until_consistent: bool = False
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
            fig.write_image(
                self.path_to_figures + "/analyse_integration_each_frame.png"
            )
            fig.write_image(
                self.path_to_figures + "/analyse_integration_each_frame.pdf"
            )
            fig.write_image(
                self.path_to_figures + "/analyse_integration_each_frame.svg"
            )
            fig.write_image(
                self.path_to_figures + "/analyse_integration_each_frame.eps"
            )
            fig.write_html(
                self.path_to_figures + "/analyse_integration_each_frame.html",
                include_mathjax="cdn",
            )

    def plot_integration_final_error(self, show: bool = True, export: bool = True):
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
            fig.write_image(self.path_to_figures + "/analyse_integration.png")
            fig.write_image(self.path_to_figures + "/analyse_integration.pdf")
            fig.write_image(self.path_to_figures + "/analyse_integration.svg")
            fig.write_image(self.path_to_figures + "/analyse_integration.eps")
            fig.write_html(
                self.path_to_figures + "/analyse_integration.html",
                include_mathjax="cdn",
            )

    def plot_obj_values(self, show: bool = True, export: bool = True):
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
            fig.write_image(self.path_to_figures + "/analyse_obj.png")
            fig.write_image(self.path_to_figures + "/analyse_obj.pdf")
            fig.write_image(self.path_to_figures + "/analyse_obj.svg")
            fig.write_image(self.path_to_figures + "/analyse_obj.eps")
            fig.write_html(
                self.path_to_figures + "/analyse_obj.html", include_mathjax="cdn"
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
        nq = self.model.nbQ()
        list_dof = list_dof = [dof.to_string() for dof in self.model.nameDof()]

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
                fig.write_image(self.path_to_figures + f"/analyse_{key}." + f)
            fig.write_html(
                self.path_to_figures + f"/analyse_{key}.html", include_mathjax="cdn"
            )

        return fig


def main():
    # path_to_files = ResultFolders.MILLER_2.value
    # model_path = Models.ACROBAT.value

    # path_to_files = ResultFolders.MILLER_2.value
    # model_path = Models.ACROBAT.value
    #
    path_to_files = (
        "/home/mickaelbegon/Documents/ipuch/dms-vs-dc-results/ACROBAT_30-08-22_2"
    )
    model_path = Models.ACROBAT.value
    export = False
    show = True

    results = ResultsAnalyse(path_to_files=path_to_files, model_path=model_path)
    results.print()
    # results.plot_time_iter(show=show, export=export, time_unit="min")
    # results.plot_obj_values(show=show, export=export)
    results.plot_integration_frame_to_frame_error(
        show=show, export=export, until_consistent=False
    )
    results.plot_integration_final_error(show=show, export=export)
    results.plot_state(
        key="q", show=show, export=export, row_col=(5, 3), until_consistent=False
    )
    results.plot_state(
        key="q_integrated",
        show=show,
        export=export,
        row_col=(5, 3),
        until_consistent=False,
    )

    # results.plot_state(key="tau", show=show, export=export, row_col=(5, 3))
    # results.plot_state(key="qddot", show=show, export=export, row_col=(5, 3))


if __name__ == "__main__":
    main()
