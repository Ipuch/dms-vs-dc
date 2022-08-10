import os, shutil
from Comparison import ComparisonAnalysis, ComparisonParameters
import pickle
from robot_leg import Humanoid2D
from humanoid_ocp import HumanoidOcp
from bioptim import OptimalControlProgram


def main():
    model_path = Humanoid2D.HUMANOID_3DOF
    dir_path = os.path.dirname(os.path.realpath(__file__))
    out_path = dir_path + "/" + model_path.name

    f = open(f"{out_path}/comp_{model_path.name}.pckl", "rb")
    comp = pickle.load(f)
    f.close()

    # comp.graphs(second_parameter="n_shooting", third_parameter="implicit_dynamics", res_path=out_path, show=True)

    df = comp.df[comp.df["implicit_dynamics"] == True]
    ocp, sol = OptimalControlProgram.load(out_path + "/" + df.iloc[0].filename + ".bo")
    # sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
