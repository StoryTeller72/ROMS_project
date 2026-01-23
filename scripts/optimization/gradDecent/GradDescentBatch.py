import mujoco
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. MODEL
# =============================
model = mujoco.MjModel.from_xml_path(
    "/home/rustam/ROMS/models/robot/robotDynamic.xml"
)

ee_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "end-effector"
)

ctrl_min = model.actuator_ctrlrange[:, 0]
ctrl_max = model.actuator_ctrlrange[:, 1]

# =============================
# 2. DATA (NPZ, batches)
# =============================
TRAJ_PATH = "/home/rustam/ROMS/data/SGDShort/trajectories.npz"
traj = np.load(TRAJ_PATH)

q_batches  = traj["q_des"]     # (N_batch, steps, n_joints)
dq_batches = traj["qd_des"]
ref_pos    = traj["ee_pos"]

N_batch, steps, n_joints = q_batches.shape
print(f"Batches={N_batch}, steps={steps}, joints={n_joints}")

# =============================
# 3. SETTINGS
# =============================
joint_indices = [0, 1, 2, 3, 4]
dim = 2 * len(joint_indices)

batch_size = 3          # <<< SGD: используем 2 батча
iterations = 30
lr = 1e-2
eps = 1e-3

# =============================
# 4. ROLLOUT + LOSS (mini-batch)
# =============================
def rollout_loss(params, batch_ids):
    total_loss = 0.0

    for b in batch_ids:
        q_ref  = q_batches[b]
        dq_ref = dq_batches[b]

        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        loss = 0.0

        for t in range(steps):
            tau = np.zeros(model.nu)

            for i, j in enumerate(joint_indices):
                Kp = params[2*i]
                Kd = params[2*i + 1]

                q  = data.qpos[j]
                dq = data.qvel[j]

                u = Kp*(q_ref[t,j]-q) + Kd*(dq_ref[t,j]-dq)
                tau[j] = np.clip(u, ctrl_min[j], ctrl_max[j])

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            loss += np.linalg.norm(
                data.xpos[ee_body_id] - ref_pos[b][t]
            )

        total_loss += loss / steps

    return total_loss / len(batch_ids)

# =============================
# 5. NUMERICAL GRADIENT (SGD)
# =============================
def numerical_gradient(params, batch_ids):
    grad = np.zeros_like(params)

    base_loss = rollout_loss(params, batch_ids)

    for i in range(len(params)):
        p1 = params.copy()
        p2 = params.copy()

        p1[i] += eps
        p2[i] -= eps

        grad[i] = (
            rollout_loss(p1, batch_ids)
            - rollout_loss(p2, batch_ids)
        ) / (2 * eps)

    return grad

# =============================
# 6. STOCHASTIC GRADIENT DESCENT
# =============================
params = np.array([
    np.random.uniform(50, 200) if i % 2 == 0 else np.random.uniform(10, 80)
    for i in range(dim)
])

loss_history = []

for it in range(iterations):
    # --- случайные 2 батча ---
    batch_ids = np.random.choice(
        N_batch, size=batch_size, replace=False
    )

    loss = rollout_loss(params, batch_ids)
    grad = numerical_gradient(params, batch_ids)

    params -= lr * grad
    loss_history.append(loss)

    print(
        f"Iter {it:02d} | "
        f"Loss={loss:.6f} | "
        f"batches={batch_ids}"
    )

# =============================
# 7. SAVE LOSS HISTORY
# =============================
with open("loss_history.txt", "w") as f:
    for i, l in enumerate(loss_history):
        f.write(f"{i}\t{l}\n")

print("\nLoss history saved to 'loss_history.txt'")

# =============================
# 8. RESULTS
# =============================
print("\n=== STOCHASTIC GD DONE ===")
for i, j in enumerate(joint_indices):
    print(
        f"Joint {j}: "
        f"Kp={params[2*i]:.2f}, "
        f"Kd={params[2*i+1]:.2f}"
    )

plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Mini-batch Loss")
plt.title("Stochastic Gradient Descent (batch size = 2)")
plt.grid(True)
plt.show()
