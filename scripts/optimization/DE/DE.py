import mujoco
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Загрузка модели
# -----------------------------
# model = mujoco.MjModel.from_xml_path("/home/rustam/ROMS/models/robot/robotDynamicW0.5.xml")
model = mujoco.MjModel.from_xml_path("/home/rustam/ROMS/models/robot/robotDynamic.xml")

# -----------------------------
# 2. Загрузка данных (NPZ с батчами)
# -----------------------------
# TRAJ_PATH = "/home/rustam/ROMS/data/sinus/noRandom/joint_trajectories_all_True.npz"
TRAJ_PATH = "/home/rustam/ROMS/data/SGDShort/trajectories.npz"
traj_data = np.load(TRAJ_PATH)
q_batches  = traj_data['q_des']   
dq_batches = traj_data['qd_des']  
ref_pos    = traj_data['ee_pos']  
N_batch, steps, n_joints = q_batches.shape

print(f"Загружено {N_batch} траекторий, {steps} шагов, {n_joints} суставов")

# -----------------------------
# 3. Настройки PD
# -----------------------------
joint_indices = [0,1,2,3,4]
num_joints = len(joint_indices)
all_iter_loss = []

ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end-effector")
ctrl_min = model.actuator_ctrlrange[:,0]
ctrl_max = model.actuator_ctrlrange[:,1]

# -----------------------------
# 4. Rollout + loss по 2 случайным батчам
# -----------------------------
# =============================
# 4. LOSS
# =============================
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

    return total_loss / N_batch

# -----------------------------
# 5. Differential Evolution с сохранением лоссов
# -----------------------------
def differential_evolution(loss_fn, bounds, pop_size=20, F=0.8, CR=0.9, generations=50, save_file="DE_loss_history.txt"):
    dim = len(bounds)
    pop = np.array([[np.random.uniform(low, high) for (low, high) in bounds] for _ in range(pop_size)])
    fitness = np.array([loss_fn(ind) for ind in pop])

    # Очистим файл перед записью
    with open(save_file, "w") as f:
        f.write("Generation,Best_Loss\n")

    for gen in range(generations):
        print(f"\nGeneration {gen+1}")
        for i in range(pop_size):
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1,r2,r3 = np.random.choice(idxs,3,replace=False)
            v = pop[r1] + F*(pop[r2]-pop[r3])

            # clip
            for j in range(dim):
                v[j] = np.clip(v[j], bounds[j][0], bounds[j][1])

            # crossover
            u = pop[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CR or j==j_rand:
                    u[j] = v[j]

            u_fitness = loss_fn(u)
            if u_fitness < fitness[i]:
                pop[i] = u
                fitness[i] = u_fitness

        best_idx = np.argmin(fitness)
        all_iter_loss.append(fitness[best_idx])
        print(f"  Best loss: {fitness[best_idx]:.6f} | Params: {pop[best_idx]}")

        # сохраняем лучший лосс поколения
        with open(save_file, "a") as f:
            f.write(f"{gen+1},{fitness[best_idx]:.6f}\n")

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx]

# -----------------------------
# 6. Определяем bounds
# -----------------------------
bounds = [
    (5, 400), (5, 300),   # joint 1
    (5, 400), (5, 250),   # joint 2
    (5, 300), (5, 200),   # joint 3
    (5, 300), (5, 120),    # joint 4
    (5, 300), (5, 120),    # joint 5
]

# -----------------------------
# 7. Запуск оптимизации
# -----------------------------
best_params, best_loss = differential_evolution(
    rollout_loss,
    bounds,
    pop_size=50,
    generations=20,
    save_file="loss_history2.txt"
)

KpList = []
KdList = []
for idx, joint in enumerate(joint_indices):
    KpList.append(best_params[2*idx])
    KdList.append(best_params[2*idx+1])

print("\n=== OPTIMIZATION DONE ===")
print("Best Kp/Kd params for joints:", best_params)
print("Final loss:", best_loss)
print("KP:", KpList, "KD:", KdList)

# -----------------------------
# 8. График
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(np.arange(len(all_iter_loss)), all_iter_loss)
plt.xlabel("Generation")
plt.ylabel("Loss")
plt.title("Differential Evolution: Loss per Generation (stochastic 2 batches)")
plt.grid(True)
plt.show()
