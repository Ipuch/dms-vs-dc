import numpy as np
from robot_leg import Models
import biorbd
import time


for model in Models:
    if model == Models.UPPER_LIMB_XYZ_TEMPLATE:
        continue
    if model == Models.HUMANOID_10DOF:
        biorbd_model = biorbd.Model(model.value[0])
    else:
        biorbd_model = biorbd.Model(model.value)
    t_fd = []
    t_id = []
    for i in range(100000):
        Q = np.random.rand(biorbd_model.nbQ()) * 2 * np.pi - np.pi
        Qdot = np.random.rand(biorbd_model.nbQdot()) * 2 * np.pi - np.pi
        Qddot = np.random.rand(biorbd_model.nbQddot()) * 2 * np.pi - np.pi
        Tau = np.random.rand(biorbd_model.nbGeneralizedTorque()) * 2 * np.pi - np.pi
        tic = time.time()
        biorbd_model.ForwardDynamics(Q, Qdot, Tau)
        toc = time.time()
        t_fd.append(toc - tic)
        tic = time.time()
        biorbd_model.InverseDynamics(Q, Qdot, Qddot)
        toc = time.time()
        t_id.append(toc - tic)

    print(f"{model.name} Forward Dynamics: {np.mean(t_fd) * 1000} ms")
    print(f"{model.name} Inverse Dynamics: {np.mean(t_id) * 1000} ms")
