import mujoco
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. Загрузка модели
# =============================
model = mujoco.MjModel.from_xml_path(
    "/home/rustam/ROMS/models/robot/robotDynamicW1.5.xml"
)

# =============================
# 2. Загрузка данных (1 траектория)
# =============================
# positions   = np.load("/home/rustam/ROMS/data/linkall/pos/0.npy")
# control_q   = np.load("/home/rustam/ROMS/data/linkall/q/0.npy")
# control_dq  = np.load("/home/rustam/ROMS/data/linkall/dq/0.npy")
positions   = np.load("/home/rustam/ROMS/data/lissajous_3d_ee_pos.npy")
control_q   = np.load("/home/rustam/ROMS/data/lissajous_3d_q.npy")
control_dq  = np.load("/home/rustam/ROMS/data/lissajous_3d_q.npy")

steps, n_joints = control_q.shape

# =============================
# 3. Настройки
# =============================
joint_indices = [0, 1, 2, 3, 4]
num_joints = len(joint_indices)

ee_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "end-effector"
)

ctrl_min = model.actuator_ctrlrange[:, 0]
ctrl_max = model.actuator_ctrlrange[:, 1]

# =============================
# 4. Rollout + loss
# =============================
def rollout_loss(params):
    data = mujoco.MjData(model)
    data.qpos[:] = control_q[0][:5]
    data.qvel[:] = control_dq[0][:5]
    mujoco.mj_forward(model, data)

    loss = 0.0

    for t in range(steps):
        tau = np.zeros(model.nu)

        for i, joint in enumerate(joint_indices):
            Kp = params[2*i]
            Kd = params[2*i + 1]

            tau_joint = (
                Kp * (control_q[t, joint]  - data.qpos[joint]) +
                Kd * (control_dq[t, joint] - data.qvel[joint])
            )

            tau[joint] = np.clip(
                tau_joint,
                ctrl_min[joint],
                ctrl_max[joint]
            )

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        loss += np.linalg.norm(
            data.xpos[ee_body_id] - positions[t]
        )

    return loss / steps

# =============================
# 5. Численный градиент
# =============================
def numeric_gradient(params, eps=1e-3):
    grad = np.zeros_like(params)
    base_loss = rollout_loss(params)

    for i in range(len(params)):
        p_plus  = params.copy()
        p_minus = params.copy()

        p_plus[i]  += eps
        p_minus[i] -= eps

        l_plus  = rollout_loss(p_plus)
        l_minus = rollout_loss(p_minus)

        grad[i] = (l_plus - l_minus) / (2 * eps)

    return grad, base_loss

# =============================
# 6. Gradient Descent
# =============================
def gradient_descent(
    init_params,
    bounds,
    lr=1,
    iters=30,
    eps=1e-3
):
    params = init_params.copy()
    loss_log = []

    for it in range(iters):
        grad, loss = numeric_gradient(params, eps)
        loss_log.append(loss)

        params -= lr * grad

        # projection to bounds
        for i, (low, high) in enumerate(bounds):
            params[i] = np.clip(params[i], low, high)

        print(
            f"Iter {it:03d} | "
            f"loss {loss:.6f} | "
            f"|grad| {np.linalg.norm(grad):.4f}"
        )

    return params, loss_log

# =============================
# 7. Bounds + init
# =============================
bounds = [
    (200, 600), (50, 300),   # joint 1
    (700, 1400), (5, 300),   # joint 2
    (900, 1500), (5, 300),   # joint 3
    (10, 300), (5, 300),    # joint 4
    (200, 600), (5, 300),    # joint 5
]

# bestParamsNoweiths
init_params = np.array([
    402, 50,
    897, 143,
    900, 5,
    300,  5,
    200,  5
], dtype=np.float64)

Kp = [402, 897, 900, 300, 200]
Kd = [50, 143, 5, 5.0, 5.0]

# =============================
# 8. Запуск оптимизации
# =============================
best_params, loss_hist = gradient_descent(
    init_params,
    bounds,
    lr=100,
    iters=200,
    eps=1e-3
)

print("\n=== GRADIENT DESCENT DONE ===")
print("Best params:", best_params)
file_path = 'w15GradDesTunning.txt'
with open(file_path, "w") as f:
    f.write(", ".join(map(str, loss_hist)))

# =============================
# 9. График
# =============================
plt.figure(figsize=(8,5))
plt.plot(loss_hist, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Gradient Descent (numeric gradients)")
plt.grid(True)
plt.show()


