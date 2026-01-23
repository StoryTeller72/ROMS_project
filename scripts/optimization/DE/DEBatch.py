import mujoco
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Загрузка модели
# -----------------------------
model = mujoco.MjModel.from_xml_path("/home/rustam/ROMS/models/robot/robotDynamic.xml")

# -----------------------------
# 2. Загрузка данных (NPZ с батчами)
# -----------------------------
TRAJ_PATH = "/home/rustam/ROMS/data/sinus/noRandom/joint_trajectories_all_True.npz"
traj_data = np.load(TRAJ_PATH)
q_batches  = traj_data['q_des']   # shape: (N_batch, steps, n_joints)
dq_batches = traj_data['qd_des']  # shape: (N_batch, steps, n_joints)
ref_pos = traj_data['ee_pos']
N_batch, steps, n_joints = q_batches.shape

print(f"Загружено {N_batch} траекторий, {steps} шагов, {n_joints} суставов")

# -----------------------------
# 3. Настройки PD
# -----------------------------
joint_indices = [0,1,2,3,4]  # первые 5 суставов
num_joints = len(joint_indices)
all_iter_loss = []

# EE ID
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end-effector")

# Ограничения актуаторов
ctrl_min = model.actuator_ctrlrange[:,0]
ctrl_max = model.actuator_ctrlrange[:,1]

# -----------------------------
# 4. Rollout + loss по батчам
# -----------------------------
def rollout_loss(params):
    """
    params: [Kp0,Kd0, Kp1,Kd1,...] для выбранных суставов
    Усреднение по всем траекториям
    """
    loss_total = 0.0

    for batch_idx in range(N_batch):
        q_ref = q_batches[batch_idx]
        dq_ref = dq_batches[batch_idx]

        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        loss = 0.0

        for t in range(steps):
            tau = np.zeros(model.nu)

            # PD контроль для каждого выбранного сустава
            for idx, joint in enumerate(joint_indices):
                Kp = params[2*idx]
                Kd = params[2*idx+1]

                q  = data.qpos[joint]
                dq = data.qvel[joint]

                tau_joint = Kp * (q_ref[t,joint] - q) + Kd * (dq_ref[t,joint] - dq)
                tau[joint] = np.clip(tau_joint, ctrl_min[joint], ctrl_max[joint])

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            err = np.linalg.norm(data.xpos[ee_body_id] - ref_pos[batch_idx][t])  
            loss += err

        loss_total += loss / steps

    return loss_total / N_batch

# -----------------------------
# 5. Differential Evolution
# -----------------------------
def differential_evolution(loss_fn, bounds, pop_size=20, F=0.8, CR=0.9, generations=50, save_file="/home/rustam/ROMS/artifacts/loss_history.txt"):
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
            v = pop[r1] + F * (pop[r2]-pop[r3])

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

        # Сохраняем текущий лучший лосс в файл
        with open(save_file, "a") as f:
            f.write(f"{gen+1},{fitness[best_idx]:.6f}\n")

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx]

# -----------------------------
# 6. Определяем bounds
# -----------------------------
bounds = [
    (50, 400), (20, 120),   # joint 1
    (40, 350), (15, 100),   # joint 2
    (40, 300), (10, 80),    # joint 3
    (30, 200), (5, 60),     # joint 4
    (20, 150), (5, 40),     # joint 5
]

# -----------------------------
# 7. Запуск оптимизации
# -----------------------------
best_params, best_loss = differential_evolution(
    rollout_loss,
    bounds,
    pop_size=50,
    generations=20,
    save_file="loss_history.txt"
)

for idx, joint in enumerate(joint_indices):
    Kp = best_params[2*idx]
    Kd = best_params[2*idx+1]

print("\n=== OPTIMIZATION DONE ===")
print("Best Kp/Kd params for joints:", best_params)
print("Final loss:", best_loss)
print('KP', Kp)
print('Kd', Kd)

# -----------------------------
# 8. График
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(np.arange(len(all_iter_loss)), all_iter_loss)
plt.xlabel("Generation")
plt.ylabel("Loss")
plt.title("Differential Evolution: Loss per Generation")
plt.grid(True)
plt.show()
