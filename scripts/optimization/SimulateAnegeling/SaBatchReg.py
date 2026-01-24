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
TRAJ_PATH = "/home/rustam/ROMS/data/sinus/noRandom/joint_trajectories_all_True.npz"
traj = np.load(TRAJ_PATH)

q_batches  = traj["q_des"]
dq_batches = traj["qd_des"]
ref_pos    = traj["ee_pos"]

N_batch, steps, n_joints = q_batches.shape
print(f"Batches={N_batch}, steps={steps}, joints={n_joints}")

# =============================
# 3. SETTINGS
# =============================
joint_indices = [0, 1, 2, 3, 4]
dim = 2 * len(joint_indices)

# SA hyperparams
T_init = 100.0
T_min = 1.0
alpha = 0.97          # коэффициент охлаждения
iterations = 100

# Параметры ограничения для каждого Kp/Kd
bounds = [
    (10, 1000), (5, 300),   # joint 1
    (10, 1000), (5, 250),   # joint 2
    (10, 1000), (5, 200),   # joint 3
    (10, 1000), (5, 120),    # joint 4
    (10, 1000), (5, 120),    # joint 5
]

# =============================
# 4. LOSS
# =============================
# =============================
# 4. LOSS с L2-регуляризацией
# =============================
reg_lambda = 1e-10  # коэффициент регуляризации

def rollout_loss(params):
    total_loss = 0.0
    for b in range(N_batch):
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
                Kd = params[2*i+1]

                q  = data.qpos[j]
                dq = data.qvel[j]

                u = Kp*(q_ref[t,j]-q) + Kd*(dq_ref[t,j]-dq)
                tau[j] = np.clip(u, ctrl_min[j], ctrl_max[j])

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            loss += np.linalg.norm(data.xpos[ee_body_id] - ref_pos[b][t])

        total_loss += loss / steps

    loss_avg = total_loss / N_batch
    # --- добавляем L2 регуляризацию по коэффициентам ---
    reg_loss = reg_lambda * np.sum(np.square(params))
    return loss_avg + reg_loss


# =============================
# 5. INITIALIZATION
# =============================
# случайное начальное значение в пределах bounds
current_params = np.array([np.random.uniform(low, high) for (low, high) in bounds])
current_loss = rollout_loss(current_params)

best_params = current_params.copy()
best_loss = current_loss

loss_history = []

# =============================
# 6. SIMULATED ANNEALING
# =============================
T = T_init
for it in range(iterations):
    # --- propose new candidate ---
    new_params = current_params + np.random.normal(0, 5.0, size=dim)
    # clip to bounds
    for i, (low, high) in enumerate(bounds):
        new_params[i] = np.clip(new_params[i], low, high)

    new_loss = rollout_loss(new_params)

    # --- accept/reject ---
    if new_loss < current_loss:
        accept = True
    else:
        prob = np.exp(-(new_loss - current_loss)/T)
        accept = np.random.rand() < prob
        accept = bool(accept)

    if accept:
        current_params = new_params
        current_loss = new_loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = current_params.copy()

    loss_history.append(current_loss)
    T = max(T * alpha, T_min)

    print(f"Iter {it:02d} | Loss={current_loss:.6f} | Best={best_loss:.6f} | T={T:.2f}")

# =============================
# 7. SAVE LOSS HISTORY
# =============================
with open("sa_loss_history.txt", "w") as f:
    for i, l in enumerate(loss_history):
        f.write(f"{i}\t{l}\n")

print("\nLoss history saved to 'sa_loss_history.txt'")

# =============================
# 8. RESULTS
# =============================
print("\n=== SIMULATED ANNEALING DONE ===")
for i, j in enumerate(joint_indices):
    print(
        f"Joint {j}: Kp={best_params[2*i]:.2f}, "
        f"Kd={best_params[2*i+1]:.2f}"
    )

# =============================
# 9. PLOT LOSS HISTORY
# =============================
plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Simulated Annealing Loss")
plt.grid(True)
plt.show()
