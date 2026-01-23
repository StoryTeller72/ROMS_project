import mujoco
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Загрузка модели
# -----------------------------
model = mujoco.MjModel.from_xml_path("/home/rustam/ROMS/models/robot/robotDynamic.xml")

# -----------------------------
# 2. Загрузка данных
# -----------------------------
positions = np.load("/home/rustam/ROMS/data/linkall/pos/0.npy")      # EE траектория
control_q  = np.load("/home/rustam/ROMS/data/linkall/q/0.npy")       # shape: (steps, n_joints)
control_dq = np.load("/home/rustam/ROMS/data/linkall/dq/0.npy")
steps = len(control_q)

# Список индексов суставов, которые хотим оптимизировать
joint_indices = [0, 1, 2, 3, 4]   # пример: первые 5 суставов
num_joints = len(joint_indices)

all_iter_loss = []

# EE ID
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end-effector")

# -----------------------------
# 3. Rollout + loss
# -----------------------------
def rollout_loss(params):
    """
    params: [Kp0, Kd0, Kp1, Kd1, ...] для всех выбранных суставов
    """
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    loss = 0.0

    for t in range(steps):
        data.ctrl[:] = 0.0
        tau = np.zeros(model.nu)

        # PD для каждого выбранного сустава
        for idx, joint in enumerate(joint_indices):
            Kp = params[2*idx]
            Kd = params[2*idx + 1]

            q  = data.qpos[joint]
            dq = data.qvel[joint]

            tau_joint = Kp * (control_q[t, joint] - q) + Kd * (control_dq[t, joint] - dq)

            # Применяем только к соответствующему актуатору
            tau[joint] = tau_joint  # предполагаем соответствие индекса актуатора и joint

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        # Ошибка по эндэффектору
        err = np.linalg.norm(data.xpos[ee_body_id] - positions[t])
        loss += err

    return loss / steps

# -----------------------------
# 4. Differential Evolution
# -----------------------------
def differential_evolution(loss_fn, bounds, pop_size=20, F=0.8, CR=0.9, generations=50):
    dim = len(bounds)
    pop = np.array([[np.random.uniform(low, high) for (low, high) in bounds] for _ in range(pop_size)])
    fitness = np.array([loss_fn(ind) for ind in pop])

    for gen in range(generations):
        print(f"\nGeneration {gen+1}")
        for i in range(pop_size):
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            v = pop[r1] + F * (pop[r2] - pop[r3])

            # clip
            for j in range(dim):
                v[j] = np.clip(v[j], bounds[j][0], bounds[j][1])

            # crossover
            u = pop[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            u_fitness = loss_fn(u)
            if u_fitness < fitness[i]:
                pop[i] = u
                fitness[i] = u_fitness

        all_iter_loss.append(np.min(fitness))
        best_idx = np.argmin(fitness)
        print(f"  Best loss: {fitness[best_idx]:.6f} | Params: {pop[best_idx]}")

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx]

# -----------------------------
# 5. Определяем bounds для всех суставов
# -----------------------------
bounds = []
for _ in joint_indices:
    bounds.append((1.0, 500.0))  # Kp
    bounds.append((1.0, 200.0))  # Kd

# -----------------------------
# 6. Запуск оптимизации
# -----------------------------
best_params, best_loss = differential_evolution(
    rollout_loss,
    bounds,
    pop_size=50,
    generations=30
)

for idx, joint in enumerate(joint_indices):
    Kp = best_params[2*idx]
    Kd = best_params[2*idx + 1]
print('KP', Kp)
print("KD", Kd)

# -----------------------------
# 7. График
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(np.arange(len(all_iter_loss)), all_iter_loss)
plt.xlabel("Generation")
plt.ylabel("Loss")
plt.title("Differential Evolution: Loss per Generation")
plt.grid(True)
plt.show()

print("\n=== OPTIMIZATION DONE ===")
print("Best Kp/Kd params for joints:", best_params)
print("Final loss:", best_loss)
