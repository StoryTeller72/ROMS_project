import mujoco as mj
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

# ================================
# НАСТРОЙКИ
# ================================
# MODEL_PATH = "/home/rustam/ROMS/models/robot/robotDynamic.xml"
# MODEL_PATH = "/home/rustam/ROMS/models/robot/robotDynamicW1.xml"
MODEL_PATH = "/home/rustam/ROMS/models/robot/robotDynamicW1.5.xml"

# Sin
Q_TRAJ_PATH = "/home/rustam/ROMS/data/linkall/q/0.npy"
DQ_TRAJ_PATH = "/home/rustam/ROMS/data/linkall/dq/0.npy"
END_EFF_PATH = "/home/rustam/ROMS/data/linkall/pos/0.npy"


# Horizontal circle
# Q_TRAJ_PATH = "/home/rustam/ROMS/data/complicated_trajectory/circle_horizontal_q.npy"
# DQ_TRAJ_PATH = "/home/rustam/ROMS/data/complicated_trajectory/circle_horizontal_dq.npy"
# END_EFF_PATH = "/home/rustam/ROMS/data/complicated_trajectory/circle_horizontal_endeffector.npy"


# Verical circle
# Q_TRAJ_PATH = "/home/rustam/ROMS/data/complicated_trajectory/circle_q.npy"
# DQ_TRAJ_PATH = "/home/rustam/ROMS/data/complicated_trajectory/circle_dq.npy"
# END_EFF_PATH = "/home/rustam/ROMS/data/complicated_trajectory/circle_endeffector.npy"

# Lisajy

# Q_TRAJ_PATH = "/home/rustam/ROMS/data/lissajous_3d_q.npy"
# DQ_TRAJ_PATH = "/home/rustam/ROMS/data/lissajous_3d_dq.npy"
# END_EFF_PATH = "/home/rustam/ROMS/data/lissajous_3d_ee_pos.npy"



# DE basе no weight
# Kp = [400, 342, 300, 200, 150]
# Kd = [101, 15, 11, 5,  5]


# # DE 0.5 kg
# Kp = [354, 858, 564, 336, 256]
# Kd = [92.6, 38.85, 7.48, 5, 5]

# DE 1 kg
# Kp = [242, 957, 1300, 42, 490]
# Kd = [168, 5.0, 5.0, 5.0, 5.0]

# DE 1.5 kg
# Kp = [402, 897, 900, 300, 200]
# Kd = [50, 143, 5, 5.0, 5.0]

# DE + GD on 1.5 kg

Kp = [406.5, 901, 900.106, 300, 200.17]
Kd = [51.7, 117, 10.58, 6.313, 5.0]

# DE + GD on 1.5 kg 200 iter
# Kp = [401, 899, 900, 300, 200]
# Kd = [50.26, 132.55, 7.46, 5.05, 5]



# SGD
# Kp = [80, 154, 188, 97.5, 153.16]
# Kd = [37, 56, 24, 21,  64]

# SA simple best
# Kp = [251, 80, 189, 231, 56]
# Kd = [80, 79, 9, 5, 6]

# Tuned  DE by GD
# Kp = [400, 342, 300, 200, 150]
# Kd = [101, 14.9, 10.9, 5, 4.9]




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
data.qpos[:] = q_ref[0][:5]
data.qvel[:] = dq_ref[0][:5]
mj.mj_forward(model, data)

# ================================
# Получаем ограничения контроллеров из модели
# ================================
# ctrlrange shape: (nu, 2) -> [min, max]
ctrl_min = model.actuator_ctrlrange[:, 0]
ctrl_max = model.actuator_ctrlrange[:, 1]

# ================================
# VIEWER
# ================================
viewer = mj.viewer.launch_passive(model, data)

# ================================
# ЛОГИ
# ================================
ee_pos_log = []

# ================================
# ОСНОВНОЙ ЦИКЛ
# ================================

 # Рисуем точки траектории
for pos in end_eff_pos:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        break  # достигнут лимит геомов
    
    size = np.array([0.001, 0.001, 0.001], dtype=np.float64).reshape(3,1)  # радиус сферы
    pos_reshaped = pos.astype(np.float64).reshape(3,1)
    mat = np.eye(3, dtype=np.float64).flatten().reshape(9,1)
    rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32).reshape(4,1)
    
    mj.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mj.mjtGeom.mjGEOM_SPHERE,
        size,
        pos_reshaped,
        mat,
        rgba
    )
    viewer.user_scn.ngeom += 1
for t in range(T):
    # текущее состояние
    q = data.qpos.copy()
    dq = data.qvel.copy()

    # PD управление
    tau = Kp * (q_ref[t][:5] - q) + Kd * (dq_ref[t][:5] - dq)

    # Обрезаем по ограничениям actuators
    tau = np.clip(tau, ctrl_min, ctrl_max)
    # Применяем
    data.ctrl[:] = tau

    # Шаг симуляции
    mj.mj_step(model, data)

    # EE положение
    mj.mj_forward(model, data)
    ee_pos_log.append(data.xpos[ee_id].copy())

    viewer.sync()

viewer.close()
ee_pos_log = np.array(ee_pos_log)

# ================================
# Построение графиков
# ================================
steps = end_eff_pos.shape[0]
time = np.arange(steps) * dt

# 2D графики по координатам
coords = ['X', 'Y', 'Z']
colors = ['r', 'g', 'b']

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

# 3D график
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(end_eff_pos[:, 0], end_eff_pos[:, 1], end_eff_pos[:, 2], linestyle='--', color='gray', label='Reference')
ax.plot(ee_pos_log[:, 0], ee_pos_log[:, 1], ee_pos_log[:, 2], linestyle='-', color='blue', label='Actual')

errors = np.linalg.norm(ee_pos_log - end_eff_pos, axis=1)
print(np.sum(errors) * dt)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('End-Effector 3D Trajectory')
ax.legend()
ax.grid(True)
ax.view_init(elev=30, azim=45)
plt.show()

