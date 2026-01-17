import mujoco as mj
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

# ================================
# НАСТРОЙКИ
# ================================
MODEL_PATH = "/home/rustam/ROMS/models/robot/robot.xml"

Q_TRAJ_PATH = "/home/rustam/ROMS/data/complicated_trajectory/circle_horizontal_q.npy"
DQ_TRAJ_PATH = "/home/rustam/ROMS/data/complicated_trajectory/circle_horizontal_dq.npy"
END_EFF_PATH = '/home/rustam/ROMS/data/complicated_trajectory/circle_horizontal_endeffector.npy'
Kp = 0.01
Kd = 0.005

# ================================
# ЗАГРУЗКА
# ================================
model = mj.MjModel.from_xml_path(MODEL_PATH)
data = mj.MjData(model)

dt = model.opt.timestep

q_ref = np.load(Q_TRAJ_PATH)     # shape: [T, nq]
dq_ref = np.load(DQ_TRAJ_PATH)   # shape: [T, nq]
end_eff_pos = np.load(END_EFF_PATH)


T = q_ref.shape[0]

ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "end-effector")

# ================================
# ИНИЦИАЛИЗАЦИЯ
# ================================
data.qpos[:] = q_ref[0]
data.qvel[:] = dq_ref[0]
mj.mj_forward(model, data)

# ================================
# VIEWER
# ================================
viewer = mj.viewer.launch_passive(model, data)

# ================================
# ЛОГИ
# ================================
ee_error_log = []
ee_pos_log = []

# ================================
# ОСНОВНОЙ ЦИКЛ
# ================================
for t in range(T):

    # --- текущее состояние ---
    q = data.qpos.copy()
    dq = data.qvel.copy()

    # --- PD управление ---
    q_cmd = (
        q_ref[t]
        + Kp * (q_ref[t] - q)
        + Kd * (dq_ref[t] - dq)
    )

    # --- ограничение суставов ---
    for j in range(model.njnt):
        low, high = model.jnt_range[j]
        q_cmd[j] = np.clip(q_cmd[j], low, high)

    # --- передаём в position actuators ---
    data.ctrl[:] = q_cmd[:model.nu]

    # --- шаг симуляции ---
    mj.mj_step(model, data)

    # --- EE ошибка ---
    mj.mj_forward(model, data)
    ee_pos = data.xpos[ee_id].copy()

    ee_pos_log.append(ee_pos)

    viewer.sync()

viewer.close()

# ================================
# АНАЛИЗ ОШИБКИ
# ================================
ee_pos_log = np.array(ee_pos_log)
error = np.linalg.norm(ee_pos_log - end_eff_pos)
print(f'Error{error}')
# ================================
# ГРАФИК JOINT ERROR
# ================================
plt.figure(figsize=(10, 5))
plt.plot(q_ref[:, 0], label="q_ref joint1")
plt.plot(ee_pos_log[:, 0], label="real EE X")
plt.legend()
plt.grid()
plt.show()
