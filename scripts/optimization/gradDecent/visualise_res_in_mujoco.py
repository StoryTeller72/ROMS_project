import mujoco
from mujoco import viewer
import numpy as np

# -----------------------------
# 1. Загрузка модели
# -----------------------------
model = mujoco.MjModel.from_xml_path("/home/rustam/ROMS/models/robot/robotDynamic.xml")
data = mujoco.MjData(model)

joint_idx = 0

# -----------------------------
# 2. Загрузка траекторий
# -----------------------------
data_npz = np.load("/home/rustam/ROMS/data/sinus/random/joint_trajectories_0_True_iter0.npz")
q_des_batch = data_npz["q_des"]  # shape: (N, steps, num_joints)
qd_des_batch = data_npz["qd_des"]

N, steps, num_joints = q_des_batch.shape
print(f"Загружено {N} траекторий, {steps} шагов, {num_joints} суставов")

# Выбираем траекторию для визуализации
traj_idx = 0

# -----------------------------
# 3. PD параметры
# -----------------------------
Kp = 1000  # сюда подставляем оптимизированные значения
Kd = 2.72

v = viewer.launch_passive(model, data)  # просто окно

# -----------------------------
# 5. Цикл симуляции с PD
# -----------------------------
for step in range(steps):
    # PD на joint3
    tau = np.zeros(model.nu)
    tau[joint_idx] = Kp * (q_des_batch[traj_idx, step, joint_idx] - data.qpos[joint_idx]) + \
                     Kd * (qd_des_batch[traj_idx, step, joint_idx] - data.qvel[joint_idx])
    data.ctrl[:] = tau

    # Шаг симуляции
    mujoco.mj_step(model, data)

    # Обновление Viewer
    v.sync()  # синхронизация отображения

print("Симуляция завершена")
