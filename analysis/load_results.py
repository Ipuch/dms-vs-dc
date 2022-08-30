import os
import pickle
from robot_leg import Models
from enums import ResultFolders
from bioptim import OptimalControlProgram


def main():
    model_path = Models.ACROBAT.value
    result_path = (
        "/home/mickaelbegon/Documents/ipuch/dms-vs-dc-results/ACROBAT_22-08-22_2"
    )
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # out_path = dir_path + "/" + model_path.name

    # open files
    files = os.listdir(result_path)
    files.sort()

    for i, file in enumerate(files):
        if file.endswith(".bo") and i > 3:
            print(file)
            ocp, sol = OptimalControlProgram.load(result_path + "/" + file)
            # sol.graphs()
            sol.animate()


if __name__ == "__main__":
    main()
