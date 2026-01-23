import mujoco as mj
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
# ================================
# НАСТРОЙКИ
# ================================
# MODEL_PATH = "/home/rustam/ROMS/models/robot/robot.xml"
MODEL_PATH = "/home/rustam/ROMS/models/robot/34joint.xml"


Q_TRAJ_PATH = "/home/rustam/ROMS/data/link4/q/0.npy"
DQ_TRAJ_PATH = "/home/rustam/ROMS/data/link4/dq/0.npy"
END_EFF_PATH = "/home/rustam/ROMS/data/link4/pos/0.npy"
Kp = 100
Kd = 4.64, 0

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
data.qpos[:] = q_ref[0][4]
data.qvel[:] = dq_ref[0][4]
mj.mj_forward(model, data)

# ================================
# VIEWER
# ================================
viewer = mj.viewer.launch_passive(model, data)

# ================================
# ЛОГИ
# ================================
ee_error_log = []
q_los = []
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
        q_ref[t][4]
        + Kp * (q_ref[t][4] - q)
        + Kd * (dq_ref[t][4] - dq)
    )

    # --- ограничение суставов ---
    for j in range(model.njnt):
        low, high = model.jnt_range[j]
        q_cmd[j] = np.clip(q_cmd[j], low, high)

    # --- передаём в position actuators ---
    data.ctrl[:] = q_cmd[:model.nu]
    q_los.append(data.ctrl[0])
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
steps = end_eff_pos.shape[0]
dt = 0.002  # или твой timestep
time = np.arange(steps) * dt

# Координаты
coords = ['X', 'Y', 'Z']
colors = ['r', 'g', 'b']

# Создаем три графика
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i in range(3):
    axs[i].plot(time, end_eff_pos[:, i], color=colors[i], linestyle='--', label='Reference')
    axs[i].plot(time, ee_pos_log[:, i], color=colors[i], linestyle='-', label='Actual')
    axs[i].set_ylabel(f'Position {coords[i]} (m)')
    axs[i].grid(True)
    axs[i].legend()

axs[2].set_xlabel('Time (s)')
fig.suptitle('End-Effector Trajectories')

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Референсная траектория
ax.plot(
    end_eff_pos[:, 0], end_eff_pos[:, 1], end_eff_pos[:, 2],
    linestyle='--', color='gray', label='Reference'
)

# Фактическая траектория
ax.plot(
    ee_pos_log[:, 0], ee_pos_log[:, 1], ee_pos_log[:, 2],
    linestyle='-', color='blue', label='Actual'
)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('End-Effector 3D Trajectory')
ax.legend()
ax.grid(True)

# Красивый ракурс
ax.view_init(elev=30, azim=45)

plt.show()