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

def generate_traj(need_joint,use_random_start_pos, amnt_traj):
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "robot" / 'robotDynamic.xml'
    SAVE_PATH = PROJECT_ROOT / "data"
    save_path = str(SAVE_PATH)
    xml_path = str(MODEL_PATH)
    num_joints = 5
    # T = 10.0
    T = 5.0
    dt = 0.002
    steps = int(T / dt)

    A_range = [(0.7, 1.2), (0.3, 0.9), (0.3, 0.9), (0.3, 0.9), (0.3, 0.9), (0.1, 0.5)]
    w_range = [(0.3, 0.8), (0.5, 0.8), (0.5, 0.8), (0.5, 0.8), (0.5, 0.8), (0.1, 0.5)]

    # --- MuJoCo модель для FK ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ee_body_id =  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end-effector")

    t = np.linspace(0, T, steps)
    q_des_batch = np.zeros((amnt_traj, steps, num_joints))
    qd_des_batch = np.zeros((amnt_traj, steps, num_joints))
    ee_pos_batch = np.zeros((amnt_traj, steps, 3))
    for traj_idx in range(amnt_traj):
        q0 = np.zeros(num_joints)
        A = np.zeros(num_joints)
        w = np.zeros(num_joints)

        if need_joint == 'all':

            for j in range(num_joints):
                A[j] = np.random.uniform(*A_range[j])
                w[j] = np.random.uniform(*w_range[j])
        else:
            # q0[need_joint] = np.random.uniform(joint_limits_min[need_joint], joint_limits_max[need_joint])
            A[need_joint] = np.random.uniform(*A_range[need_joint])
            w[need_joint] = np.random.uniform(*w_range[need_joint])

        for k in range(steps):
            q_des_batch[traj_idx][k]  = q0 + A * np.sin(w * t[k])
            qd_des_batch[traj_idx][k] = A * w * np.cos(w * t[k])

            # ---- FK для эндэффектора ----
            data.qpos = q_des_batch[traj_idx][k]
            mujoco.mj_forward(model, data)
            ee_pos_batch[traj_idx][k] = data.xpos[ee_body_id].copy()
        visualisae(q_des_batch[traj_idx])
    # np.savez(f"/home/rustam/ROMS/data/sinus/noRandom/joint_trajectories_{need_joint}_{use_random_start_pos}.npz", q_des=q_des_batch, qd_des=qd_des_batch, ee_pos=ee_pos_batch)
    np.savez(f"/home/rustam/ROMS/data/SGDShort/trajectories.npz", q_des=q_des_batch, qd_des=qd_des_batch, ee_pos=ee_pos_batch)
    print("Сгенерировано и сохранено:", q_des_batch.shape)


if __name__ == '__main__':
   generate_traj('all', True,10)