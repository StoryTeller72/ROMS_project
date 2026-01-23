import mujoco
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. MODEL
# =============================
model = mujoco.MjModel.from_xml_path("/home/rustam/ROMS/models/robot/robotDynamic.xml")
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end-effector")
ctrl_min = model.actuator_ctrlrange[:, 0]
ctrl_max = model.actuator_ctrlrange[:, 1]

# =============================
# 2. DATA
# =============================
TRAJ_PATH = "/home/rustam/ROMS/data/SGDShort/trajectories.npz"
traj = np.load(TRAJ_PATH)
q_batches  = traj["q_des"]
dq_batches = traj["qd_des"]
ref_pos    = traj["ee_pos"]

N_batch, steps, n_joints = q_batches.shape
print(f"Batches={N_batch}, steps={steps}, joints={n_joints}")

# =============================
# 3. SETTINGS
# =============================
joint_indices = [0,1,2,3,4]
dim = 2 * len(joint_indices)
batch_size = 2
iterations = 30
lr = 1e-2
eps = 1e-3

payload_mass = 0.5  # вес груза на ендэффекторе
reg_lambda = 1e-4   # коэффициент L2-регуляризации

# =============================
# 4. ROLLOUT + LOSS
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
            for i,j in enumerate(joint_indices):
                Kp = params[2*i]
                Kd = params[2*i+1]
                q  = data.qpos[j]
                dq = data.qvel[j]
                u = Kp*(q_ref[t,j]-q) + Kd*(dq_ref[t,j]-dq)
                tau[j] = np.clip(u, ctrl_min[j], ctrl_max[j])
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            err = np.linalg.norm(data.xpos[ee_body_id] - ref_pos[b][t])
            loss += err * payload_mass  # весовой коэффициент

        total_loss += loss / steps

    loss_avg = total_loss / len(batch_ids)
    # --- L2 регуляризация по Kp/Kd ---
    reg_loss = reg_lambda * np.sum(np.square(params))
    return loss_avg + reg_loss
