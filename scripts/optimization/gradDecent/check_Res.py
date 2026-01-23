import mujoco
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Загрузка модели
# -----------------------------
model = mujoco.MjModel.from_xml_path("/home/rustam/ROMS/models/robot/robotDynamic.xml")
data = mujoco.MjData(model)
joint_idx = 1

# -----------------------------
# 2. Загрузка траекторий
# -----------------------------
data_npz = np.load("/home/rustam/ROMS/data/sinus/joint_trajectories_1_False.npz")
q_des_batch = data_npz["q_des"]  # shape: (N, steps, num_joints)
qd_des_batch = data_npz["qd_des"]

N, steps, num_joints = q_des_batch.shape
print(f"Загружено {N} траекторий, {steps} шагов, {num_joints} суставов")

# -----------------------------
# 3. Параметры PD для проверки
# -----------------------------
Kp = 10.968  # сюда подставляем оптимизированные значения
Kd = 2.72

# -----------------------------
# 4. Rollout и сохранение позиций
# -----------------------------
q_actual_batch = np.zeros((N, steps))

for traj_idx in range(N):
    data_copy = mujoco.MjData(model)
    data_copy.qpos[:] = data.qpos[:]
    data_copy.qvel[:] = data.qvel[:]

    for k in range(steps):
        q = data_copy.qpos
        qd = data_copy.qvel

        # PD на одном суставе
        tau = np.zeros(model.nu)
        tau[joint_idx] = Kp * (q_des_batch[traj_idx, k, joint_idx] - q[joint_idx]) \
                        + Kd * (qd_des_batch[traj_idx, k, joint_idx] - qd[joint_idx])
        data_copy.ctrl[:] = tau

        mujoco.mj_step(model, data_copy)

        # Сохраняем реальную позицию сустава
        q_actual_batch[traj_idx, k] = q[joint_idx]

# -----------------------------
# 5. Визуализация
# -----------------------------
for traj_idx in range(N):
    plt.figure()
    plt.plot(q_des_batch[traj_idx, :, joint_idx], label="Target q")
    plt.plot(q_actual_batch[traj_idx, :], label="Actual q")
    plt.xlabel("Step")
    plt.ylabel(f"Joint {joint_idx} position [rad]")
    plt.title(f"Trajectory {traj_idx}")
    plt.legend()
    plt.show()
