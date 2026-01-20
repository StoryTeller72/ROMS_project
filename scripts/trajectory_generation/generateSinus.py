import mujoco.viewer
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from pathlib import Path
import random

def visualisae(q_traj):
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "robot" / 'robotDynamic.xml'
    xml_path = str(MODEL_PATH)
    # -------------------------------
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[:] = q_traj[0]
    mujoco.mj_forward(model, data)
    viewer = mujoco.viewer.launch_passive(model, data)
    for i in range(len(q_traj)):
        data.qpos[:] = q_traj[i][:]  # двигаем только один сустав
        mujoco.mj_forward(model, data)  # без mj_step для идеального движени

        viewer.sync()

    viewer.close()

def generate_traj(need_joint,use_random_start_pos):
    num_joints = 6   # например, 6 суставов
    N = 5            # количество траекторий
    T = 5.0          # длительность траектории в секундах
    dt = 0.002
    steps = int(T / dt)

    # Амплитуда и частота синусоид
    A_range = [(0.2, 1), (0.1, 0.6), (0.1, 1), (0.1, 1), (0.1, 0.5), (0.1, 0.5)]
    w_range = [(0.2, 0.5), (0.1, 0.5), (0.1, 0.5), (0.1, 0.5), (0.1, 0.5), (0.1, 0.5)]


    # Ограничения суставов (можно получить из model.jnt_range)
    joint_limits_min = np.array([-np.pi/2, -np.pi/4, -np.pi/4, -np.pi/4, -np.pi/4, -np.pi/4])
    joint_limits_max = np.array([ np.pi/2,  np.pi/4,  np.pi/4,  np.pi/4,  np.pi/4,  np.pi/4])

    q_des_batch = np.zeros((N, steps, num_joints))
    qd_des_batch = np.zeros((N, steps, num_joints))

    t = np.linspace(0, T, steps)

    for traj_idx in range(N):
        # Случайные начальные углы в пределах joint_limits
        if use_random_start_pos:
            q0 = np.random.uniform(joint_limits_min, joint_limits_max)
        else:
            q0 = np.zeros(6)
        # Случайные амплитуды и частоты для каждого сустава
        A = np.zeros(6)
        w = np.zeros(6)

        if need_joint == 'all':
            for joint_id in range(6):
                A_min, A_max = A_range[joint_id]
                w_min, w_max = w_range[joint_id]
                A[joint_id] =  np.random.uniform(A_min, A_max)
                w[joint_id] = np.random.uniform(w_min, w_max)
        else:
            A_min, A_max = A_range[need_joint]
            w_min, w_max = w_range[need_joint]
            A[need_joint] =  np.random.uniform(A_min, A_max)
            w[need_joint] = np.random.uniform(w_min, w_max)


        for j in range(num_joints):
            q_des_batch[traj_idx, :, j] = q0[j] + A[j] * np.sin(w[j] * t)
            qd_des_batch[traj_idx, :, j] = A[j] * w[j] * np.cos(w[j] * t)

    # -----------------------------
    # Сохраняем в npz
    # -----------------------------

    visualisae(q_des_batch[0,:,:])
    np.savez(f"/home/rustam/ROMS/data/sinus/noRandom/joint_trajectories_{need_joint}_{use_random_start_pos}.npz", q_des=q_des_batch, qd_des=qd_des_batch)

    print("Сгенерировано и сохранено:", q_des_batch.shape)


if __name__ == '__main__':
    for i in range(5):
        generate_traj(i, False)