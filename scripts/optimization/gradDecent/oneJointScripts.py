import mujoco
import numpy as np

# -----------------------------
# 1. Загрузка модели
# -----------------------------
model = mujoco.MjModel.from_xml_path("/home/rustam/ROMS/models/robot/robotDynamic.xml")
data = mujoco.MjData(model)
joint_idx = 1

data_npz = np.load("/home/rustam/ROMS/data/sinus/joint_trajectories_1_False.npz")
q_des_batch = data_npz["q_des"]  # shape: (N, steps, num_joints)
qd_des_batch = data_npz["qd_des"]

N, steps, num_joints = q_des_batch.shape
print(f"Загружено {N} траекторий, {steps} шагов, {num_joints} суставов")

# Сохраняем текущее положение остальных суставов
q0 = data.qpos[6:].copy()

# -----------------------------
# 2. Генерация нескольких синусоидальных траекторий
# -----------------------------
N = 5           # количество траекторий
T = 2.0         # длительность
dt = model.opt.timestep
# -----------------------------
# 3. Rollout функция для одного сустава
# -----------------------------
def rollout_loss(Kp, Kd, traj_idx):
    loss = 0.0
    data_copy = mujoco.MjData(model)
    data_copy.qpos[:] = data.qpos[:]
    data_copy.qvel[:] = data.qvel[:]

    for k in range(steps):
        q = data_copy.qpos
        qd = data_copy.qvel

        tau = np.zeros(model.nu)
        tau[joint_idx] = Kp * (q_des_batch[traj_idx, k, joint_idx] - q[joint_idx]) \
                        + Kd * (qd_des_batch[traj_idx, k, joint_idx] - qd[joint_idx])
        data_copy.ctrl[:] = tau

        mujoco.mj_step(model, data_copy)

        loss += (q[joint_idx] - q_des_batch[traj_idx, k, joint_idx])**2 * model.opt.timestep
    return loss


# -----------------------------
# 4. Batch loss
# -----------------------------
def total_loss(Kp, Kd):
    return np.mean([rollout_loss(Kp, Kd, i) for i in range(N)])

# -----------------------------
# 5. Численный градиент
# -----------------------------
def compute_grad(Kp, Kd, eps=1e-3):
    L = total_loss(Kp, Kd)
    dKp = (total_loss(Kp + eps, Kd) - L) / eps
    dKd = (total_loss(Kp, Kd + eps) - L) / eps
    return np.array([dKp, dKd])

# -----------------------------
# 6. Оптимизация градиентным спуском
# -----------------------------
params = np.array([10.0, 1.0])  # начальные Kp, Kd
lr = 0.1
num_steps = 20

for step in range(num_steps):
    grads = compute_grad(params[0], params[1])
    params -= lr * grads
    loss_val = total_loss(params[0], params[1])
    print(f"Step {step}: Kp={params[0]:.3f}, Kd={params[1]:.3f}, loss={loss_val:.5f}")

print("Оптимизация завершена!")
print(f"Лучшие параметры: Kp={params[0]:.3f}, Kd={params[1]:.3f}")