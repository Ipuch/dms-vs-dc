from typing import Any, Union, Tuple
import numpy as np
from numpy import ndarray

from plotly import graph_objects as go
import plotly.express as px
from pandas import DataFrame

import biorbd
from bioptim import Solution


def compute_error_single_shooting(
    time: Union[np.ndarray, list],
    n_shooting: int,
    model: biorbd.Model,
    q: np.ndarray,
    q_integrated: np.ndarray,
    duration: float = None,
):
    """
    Compute the error between the solution of the OCP and the solution of the integrated OCP

    Parameters
    ----------
    time : np.ndarray
        Time vector
    n_shooting : int
        Number of shooting points
    model : biorbd.Model
        Model
    q : np.ndarray
        ocp generalized coordinates
    q_integrated : np.ndarray
        integrated generalized coordinates
    duration: float
        The duration to report the error in states btween the two solutions

    Returns
    -------
    The error between the two solutions
    :tuple
    """

    duration = time[-1] if duration is None else duration

    if time[-1] < duration:
        raise ValueError(
            f"Single shooting integration duration must be smaller than ocp duration :{time[-1]} s"
        )

    # get the index of translation and rotation dof
    trans_idx, rot_idx = get_trans_and_rot_idx(model=model)

    sn_1s = int(n_shooting / time[-1] * duration)  # shooting node at {duration} second
    single_shoot_error_r = (
        rmse(q[rot_idx, sn_1s], q_integrated[rot_idx, sn_1s]) * 180 / np.pi
        if len(rot_idx) > 0
        else np.nan
    )

    single_shoot_error_t = (
        (rmse(q[trans_idx, sn_1s], q_integrated[trans_idx, sn_1s]) / 1000)
        if len(trans_idx) > 0
        else np.nan
    )

    return (
        single_shoot_error_t,
        single_shoot_error_r,
    )


def compute_error_single_shooting_each_frame(
    time: Union[np.ndarray, list],
    n_shooting: int,
    model: biorbd.Model,
    q: np.ndarray,
    q_integrated: np.ndarray,
) -> tuple[ndarray, ndarray]:
    """
    This function compute the error between the solution of the OCP and the solution of the integrated OCP for each frame

    Parameters
    ----------
    time : np.ndarray
        Time vector
    n_shooting : int
        Number of shooting points
    model : biorbd.Model
        Model
    q : np.ndarray
        ocp generalized coordinates
    q_integrated : np.ndarray
        integrated generalized coordinates
    duration: float
        The duration to report the error in states btween the two solutions

    Returns
    -------
    The error between the two solutions for each frame
    """

    single_shoot_error_t = np.zeros(time.shape[0])
    single_shoot_error_r = np.zeros(time.shape[0])

    for i, t in enumerate(time):
        (
            single_shoot_error_t[i],
            single_shoot_error_r[i],
        ) = compute_error_single_shooting(
            time=time,
            n_shooting=n_shooting,
            model=model,
            q=q,
            q_integrated=q_integrated,
            duration=t,
        )

    return single_shoot_error_t, single_shoot_error_r


def get_trans_and_rot_idx(model: biorbd.Model) -> tuple:
    """
    This function

    Parameters
    ----------
    model: biorbd.Model
        The model from biorbd

    Returns
    ----------
    The index of the translation and rotation dof
    """
    # get the index of translation and rotation dof
    trans_idx = []
    rot_idx = []
    for i in range(model.nbQ()):
        if model.nameDof()[i].to_string()[-4:-1] == "Rot":
            rot_idx += [i]
        elif "_Trans" in model.nameDof()[i].to_string():
            trans_idx += [i]
        else:
            ValueError("This type is not recognized")
    rot_idx = np.array(rot_idx)
    trans_idx = np.array(trans_idx)
    return trans_idx, rot_idx


def stack_states(states: list[dict], key: str = "q"):
    """
    Stack the controls in one vector

    Parameters
    ----------
    states : list[dict]
        List of dictionaries containing the states
    key : str
        Key of the states to stack such as "q" or "qdot"
    """
    the_tuple = [
        s[key][:, :] if i == len(states) - 1 else s[key][:, :-1]
        for i, s in enumerate(states)
    ]
    return np.hstack(the_tuple)


def stack_controls(controls: list[dict], key: str = "tau"):
    """
    Stack the controls in one vector

    Parameters
    ----------
    controls : list[dict]
        List of dictionaries containing the controls
    key : str
        Key of the controls to stack such as "tau" or "qddot"
    """
    the_tuple = (
        c[key][:, :-1] if i < len(controls) else c[key][:, :]
        for i, c in enumerate(controls)
    )
    return np.hstack(the_tuple)


def define_time(time: list, n_shooting: tuple):
    """
    Create the time vector

    Parameters
    ----------
    time : list
        List of duration of each phase of the simulation
    n_shooting : tuple
        Number of shooting points for each phase
    """
    the_tuple = (
        np.linspace(
            0, float(time[i]) - 1 / n_shooting[i] * float(time[i]), n_shooting[i]
        )
        if i < len(time)
        else np.linspace(
            float(time[i]), float(time[i]) + float(time[i + 1]), n_shooting[i] + 1
        )
        for i, t in enumerate(time)
    )
    return np.hstack(the_tuple)


def rmse(predictions, targets) -> float:
    """
    Compute the Root Mean Square Error

    Parameters
    ----------
    predictions : numpy.array
        Predictions
    targets : numpy.array
        Targets

    Returns
    -------
    rmse : float
        Root Mean Square Error
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def my_traces(
    fig: go.Figure,
    dyn: str,
    grps: list,
    df: DataFrame,
    key: str,
    row: int,
    col: int,
    ylabel: str = None,
    title_str: str = None,
    ylog: bool = True,
    color: list = None,
    show_legend: bool = False,
):
    """
    This function is used to boxplot the data in the dataframe.

    Parameters
    ----------
    fig : go.Figure
        The figure to which the boxplot is added.
    dyn : str
        The name of the dynamic system.
    grps : list
        The list of groups to be plotted.
    df : DataFrame
        The dataframe containing the data.
    key : str
        The key of the dataframe such as "q" or "tau".
    row : int
        The row of the subplot.
    col : int
        The column of the subplot.
    ylabel : str
        The label of the y-axis.
    title_str : str
        The title of the subplot.
    ylog : bool
        If true, the y-axis is logarithmic.
    color : list
        The colors of the boxplot.
    show_legend : bool
        If true, the legend is shown.
    """
    ylog = "log" if ylog == True else None
    if (col == 1 and row == 1) or (col is None or row is None) or show_legend == True:
        showleg = True
    else:
        showleg = False

    for ii, d in enumerate(dyn):
        # manage color
        c = (
            px.colors.hex_to_rgb(px.colors.qualitative.D3[ii % 9])
            if color is None
            else color[ii]
        )
        c = str(f"rgba({c[0]},{c[1]},{c[2]},0.5)")
        c1 = (
            px.colors.qualitative.D3[ii % 9]
            if color is None
            else px.colors.label_rgb(color[ii])
        )
        fig.add_trace(
            go.Box(
                x=df["grps"][df["grps"] == d],
                y=df[key][df["grps"] == d],
                name=d,
                boxpoints="all",
                width=0.4,
                pointpos=-2,
                legendgroup=grps[ii],
                fillcolor=c,
                marker=dict(opacity=0.5),
                line=dict(color=c1),
            ),
            row=row,
            col=col,
        )

    fig.update_traces(
        jitter=0.8,  # add some jitter on points for better visibility
        marker=dict(size=3),
        row=row,
        col=col,
        showlegend=showleg,
        selector=dict(type="box"),
    )
    fig.update_yaxes(
        type=ylog,
        row=row,
        col=col,
        title=ylabel,
        title_standoff=2,
        exponentformat="e",
    )
    fig.update_xaxes(
        row=row,
        col=col,
        color="black",
        showticklabels=False,
        ticks="",
    )
    return fig


def add_annotation_letter(
    fig: go.Figure,
    letter: str,
    x: float,
    y: float,
    row: int = None,
    col: int = None,
    on_paper: bool = False,
) -> go.Figure:
    """
    Adds a letter to the plot for scientific articles.

    Parameters
    ----------
    fig: go.Figure
        The figure to annotate
    letter: str
        The letter to add to the plot.
    x: float
        The x coordinate of the letter.
    y: float
        The y coordinate of the letter.
    row: int
        The row of the plot to annotate.
    col: int
        The column of the plot to annotate.
    on_paper: bool
        If True, the annotation will be on the paper instead of the axes
    Returns
    -------
    The figure with the letter added.
    """
    if on_paper:
        xref = "paper"
        yref = "paper"
    else:
        xref = f"x{col}" if row is not None else None
        yref = f"y{row}" if row is not None else None

    fig["layout"]["annotations"] += (
        dict(
            x=x,
            y=y,
            xanchor="left",
            yanchor="bottom",
            text=f"{letter})",
            font=dict(family="Times", size=14, color="black"),
            showarrow=False,
            xref=xref,
            yref=yref,
        ),
    )

    return fig


def generate_windows_size(nb: int) -> tuple:
    """
    Defines the number of column and rows of subplots from the number of variables to plot.

    Parameters
    ----------
    nb: int
        Number of variables to plot

    Returns
    -------
    The optimized number of rows and columns
    """

    n_rows = int(round(np.sqrt(nb)))
    return n_rows + 1 if n_rows * n_rows < nb else n_rows, n_rows


def plot_all_dof(
    fig: go.Figure,
    key: str,
    df: DataFrame,
    list_dof: list,
    idx_rows: list,
    idx_cols: list,
    trans_idx: list,
    rot_idx: list,
    until_consistent: bool = None,
):
    """
    This function plots all generalized coordinates and all torques for all MillerDynamics
    contained in the main cluster of optimal costs

    Parameters
    ----------
    fig : go.Figure
        Figure to plot on
    key : str
        Key of the dataframe to plot (q, qdot, tau, muscle)
    df : DataFrame
        Dataframe of all results
    list_dof : list
        List of all dofs
    idx_rows : list
        List of the rows to plot
    idx_cols : list
        List of the columns to plot
    trans_idx: list
        List of translation dof indexes
    rot_idx: mlist
        List of rotation dof indexes

    Returns
    -------
    fig : go.Figure
        Figure with all plots
    """

    showleg = False

    for i_dof, dof_name in enumerate(list_dof):
        idx_row = idx_rows[i_dof]
        idx_col = idx_cols[i_dof]
        for _, row in df.iterrows():

            # rotations in degrees
            coef = 180 / np.pi if i_dof in rot_idx and "q" in key else 1

            idx_end = (
                int(row.consistent_threshold)
                if until_consistent
                else int(len(row.time))
            )
            print(idx_end)
            time = row.time[:idx_end]
            y = row[key][i_dof][:idx_end] * coef

            fig.add_scatter(
                x=time,
                y=y,
                mode="lines",
                marker=dict(
                    size=2,
                    color=px.colors.qualitative.D3[row.grp_number],
                ),
                line=dict(width=1.5, color=px.colors.qualitative.D3[row.grp_number]),
                name=row.grps if row.irand == 0 and i_dof == 0 else None,
                legendgroup=row.grps,
                showlegend=True if row.irand == 0 and i_dof == 0 else False,
                row=idx_row,
                col=idx_col,
            )
            fig.update_yaxes(tickfont=dict(size=11))

    fig.update_layout(
        height=1200,
        width=1200,
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        legend=dict(
            title_font_family="Times New Roman",
            font=dict(family="Times New Roman", color="black", size=18),
            orientation="h",
            yanchor="bottom",
            y=1.05,
            x=0.5,
            xanchor="center",
            valign="top",
        ),
        font=dict(
            size=19,
            family="Times New Roman",
        ),
        yaxis=dict(color="black"),
        template="simple_white",
        boxgap=0.2,
    )
    return fig
