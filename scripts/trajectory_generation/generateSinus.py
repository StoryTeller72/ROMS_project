import mujoco as mj
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_traj(need_joint, amnt_traj):
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "robot" / 'robotDynamic.xml'
    SAVE_PATH = PROJECT_ROOT / "data"
    save_path = str(SAVE_PATH)
    xml_path = str(MODEL_PATH)
    num_joints = 6
    T = 10.0
    dt = 0.002
    steps = int(T / dt)

    A_range = [(0.7, 1.2), (0.3, 0.9), (0.3, 0.9), (0.3, 0.9), (0.3, 0.9), (0.1, 0.5)]
    w_range = [(0.3, 0.8), (0.5, 0.8), (0.5, 0.8), (0.5, 0.8), (0.5, 0.8), (0.1, 0.5)]

    joint_limits_min = np.array([-np.pi/2]*6)
    joint_limits_max = np.array([0]*6)


    # --- MuJoCo модель для FK ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ee_body_id =  mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "end-effector")

    t = np.linspace(0, T, steps)
    q_des_batch  = np.zeros((steps, num_joints))
    qd_des_batch = np.zeros((steps, num_joints))
    ee_pos_batch = []   
    for traj_idx in range(amnt_traj):
        q0 = np.zeros(6)
        A = np.zeros(6)
        w = np.zeros(6)

        if need_joint == 'all':

            for j in range(6):
                A[j] = np.random.uniform(*A_range[j])
                w[j] = np.random.uniform(*w_range[j])
        else:
            # q0[need_joint] = np.random.uniform(joint_limits_min[need_joint], joint_limits_max[need_joint])
            A[need_joint] = np.random.uniform(*A_range[need_joint])
            w[need_joint] = np.random.uniform(*w_range[need_joint])

        for k in range(steps):
            q_des_batch[k]  = q0 + A * np.sin(w * t[k])
            qd_des_batch[k] = A * w * np.cos(w * t[k])

            # ---- FK для эндэффектора ----
            data.qpos = q_des_batch[k]
            mujoco.mj_forward(model, data)
            ee_pos_batch.append(data.xpos[ee_body_id].copy())
        ee_positions = np.array(ee_pos_batch)
        np.save(save_path + f'/link{need_joint}/pos/{traj_idx}.npy', ee_positions)
        np.save(save_path + f'/link{need_joint}/q/{traj_idx}.npy', q_des_batch)
        np.save(save_path + f'/link{need_joint}/dq/{traj_idx}.npy', qd_des_batch)
        print(f"Координаты эндэффектора сохранены в {save_path}")

    print("Сохранено:")


if __name__ == '__main__':
    # for i in range(6):
    generate_traj(4, 1)
