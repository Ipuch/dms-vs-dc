"""
This script is reading and organizing the raw data results from Miller Optimal control problems into a nice DataFrame.
It requires the all the raw data to run the script.
"""
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

import biorbd
from bioptim import OptimalControlProgram, Shooting

from utils import (
    stack_states,
    stack_controls,
    define_time,
    compute_error_single_shooting,
)

out_path_raw = "../../dms-vs-dc-results/raw_28-07-22"
model_path = "../robot_leg/models/hexapod_leg.bioMod"

# open files
files = os.listdir(out_path_raw)
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
]
df_results = pd.DataFrame(columns=column_names)

for i, file in enumerate(files):
    if file.endswith(".pckl"):
        print(file)
        p = Path(f"{out_path_raw}/{file}")
        file_path = open(p, "rb")
        data = pickle.load(file_path)

        # DM to array
        data["cost"] = np.array(data["cost"])[0][0]

        # compute error
        model = biorbd.Model(model_path)

        data["translation_error"], data["rotation_error"] = compute_error_single_shooting(
            model=model,
            n_shooting=data["n_shooting"],
            time=np.array(data["time"]),
            q=data["q"],
            q_integrated=data["q_integrated"],
        )

        data["translation_error_2"], data["rotation_error_2"] = compute_error_single_shooting(
            model=model,
            n_shooting=data["n_shooting"],
            time=np.array(data["time_linear"]),
            q=data["q"],
            q_integrated=data["q_integrated_linear"],
        )

        print(data["q"].shape)
        print(data["q_integrated"].shape)

        data["grps"] = data["ode_solver"].__str__() + data["defects_type"].value

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

print(df_results[["dynamics_type", "n_shooting", "ode_solver", "translation_error", "rotation_error", "translation_error_2", "rotation_error_2"]])
df_results[["dynamics_type", "ode_solver", "status", "translation_error", "rotation_error", "translation_error_2", "rotation_error_2"]].to_csv(f"{out_path_raw}/results.csv")
# print(df_results[["ode_solver", ]])

# fill new columns
# n_row = len(df_results)
# df_results["t"] = None
# df_results["t_integrated"] = None
# df_results["tau_integrated"] = None
# df_results["q"] = None
# df_results["qdot"] = None
# df_results["qddot"] = None
# df_results["tau"] = None
# df_results["int_T"] = None
# df_results["int_R"] = None
# df_results["angular_momentum"] = None
# df_results["linear_momentum"] = None
# df_results["comdot"] = None
# df_results["comddot"] = None
# df_results["angular_momentum_rmse"] = None
# df_results["linear_momentum_rmse"] = None
# df_results["T1"] = None
# df_results["T2"] = None

# m = biorbd.Model(model.value[0])
# n_step = 5
# N = 2
# N_integrated = 2
# for index, row in df_results.iterrows():
#     print(index)
#
#     q_integrated = row.q_integrated
#     qdot_integrated = row.qdot_integrated
#
#     # non integrated values, at nodes.
#     t = define_time([0, 0.3], [row.n_shooting])
#     N = len(t)
#     q = stack_states(row.states, "q")
#     qdot = stack_states(row.states, "qdot")
#     tau = stack_controls(row.controls, "tau")
#
#     # compute qddot
#     # if row.rigidbody_dynamics == XX:
#     #     qddot = stack_controls(row.controls, "qddot")
#     #
#     # elif row.rigidbody_dynamics == XX:
#     #     qddot = np.zeros((m.nbQ(), N))
#     #     for ii in range(N):
#     #         qddot[:, ii] = m.ForwardDynamicsConstraintsDirect(q[:, ii], qdot[:, ii], tau[:, ii]).to_array()
#     #
#     # elif (
#     #     row.rigidbody_dynamics == XX:
#     # ):
#     #     qddot = stack_states(row.states, "qddot")
#
#     df_results.at[index, "t"] = t
#     df_results.at[index, "q"] = q
#     df_results.at[index, "qdot"] = qdot
#     # df_results.at[index, "qddot"] = qddot
#     df_results.at[index, "tau"] = tau

# EXTRA COMPUTATIONS
# NICE LATEX LABELS
# df_results["dynamics_type_label"] = None
# df_results.loc[df_results["dynamics_type"] == MillerDynamics.EXPLICIT, "dynamics_type_label"] = r"$\text{Full-Exp}$"
# df_results.loc[
#     df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT, "dynamics_type_label"
# ] = r"$\text{Base-Exp}$"
# df_results.loc[
#     df_results["dynamics_type"] == MillerDynamics.IMPLICIT, "dynamics_type_label"
# ] = r"$\text{Full-Imp-}\ddot{q}$"
# df_results.loc[
#     df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT, "dynamics_type_label"
# ] = r"$\text{Base-Imp-}\ddot{q}$"
# df_results.loc[
#     df_results["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, "dynamics_type_label"
# ] = r"$\text{Full-Imp-}\dddot{q}$"
# df_results.loc[
#     df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT, "dynamics_type_label"
# ] = r"$\text{Base-Imp-}\dddot{q}$"

# COST FUNCTIONS
# # four first functions for each phase
# df_results["cost_J"] = None
# # df_results["cost_angular_momentum"] = None
#
# for ii in range(10):
#     df_results[f"cost_J{ii}"] = None
#
# for index, row in df_results.iterrows():
#     print(index)
#     dc = row.detailed_cost
#
#     df_results.at[index, "cost_J"] = np.sum([row.detailed_cost[idx]["cost_value_weighted"] for idx in idx_J])
#     print(np.sum([row.detailed_cost[idx]["cost_value_weighted"] for idx in idx_J]))
#
#     df_results.at[index, "cost_angular_momentum"] = np.sum(
#         [row.detailed_cost[idx]["cost_value_weighted"] for idx in idx_angular_momentum]
#     )

# df_results["grps"] = None
# df_results.loc[df_results["dynamics_type"] == MillerDynamics.EXPLICIT, "grps"] = "Explicit"
# df_results.loc[df_results["dynamics_type"] == MillerDynamics.ROOT_EXPLICIT, "grps"] = "Root_Explicit"
# df_results.loc[df_results["dynamics_type"] == MillerDynamics.IMPLICIT, "grps"] = "Implicit_qddot"
# df_results.loc[df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT, "grps"] = "Root_Implicit_qddot"
# df_results.loc[df_results["dynamics_type"] == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT, "grps"] = "Implicit_qdddot"
# df_results.loc[df_results["dynamics_type"] == MillerDynamics.ROOT_IMPLICIT_QDDDOT, "grps"] = "Root_Implicit_qdddot"

# df_results = pd.read_pickle("Dataframe_results_metrics_5.pkl")
# df_results.to_pickle("Dataframe_results_metrics_5.pkl")

# COMPUTE CLUSTERS of same value for Cost_J
# df_results["main_cluster"] = False
# # specify the value for each dynamic type
# cost_J_cluster_values = [10.63299, 10.61718, 2.68905, 2.591474, 10.58839, 10.58604]
# for index, row in df_results.iterrows():
#     if row.dynamics_type == MillerDynamics.EXPLICIT:
#         cluster_val = cost_J_cluster_values[0]
#     elif row.dynamics_type == MillerDynamics.ROOT_EXPLICIT:
#         cluster_val = cost_J_cluster_values[1]
#     elif row.dynamics_type == MillerDynamics.IMPLICIT:
#         cluster_val = cost_J_cluster_values[2]
#     elif row.dynamics_type == MillerDynamics.ROOT_IMPLICIT:
#         cluster_val = cost_J_cluster_values[3]
#     elif row.dynamics_type == MillerDynamics.IMPLICIT_TAU_DRIVEN_QDDDOT:
#         cluster_val = cost_J_cluster_values[4]
#     elif row.dynamics_type == MillerDynamics.ROOT_IMPLICIT_QDDDOT:
#         cluster_val = cost_J_cluster_values[5]
#
#     if abs(cluster_val - row["cost_J"]) < 1e-3:
#         print(row.dynamics_type)
#         print(cluster_val - row["cost_J"])
#         print(row.irand)
#         df_results.at[index, "main_cluster"] = True

# saves the dataframe
df_results.to_pickle("Dataframe_results_metrics.pkl")
