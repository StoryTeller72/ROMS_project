import mujoco as mj
import numpy as np
from pathlib import Path

# ==============================
# Загрузка модели
# ==============================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "robot" / "robot.xml"

model = mj.MjModel.from_xml_path(str(MODEL_PATH))
data = mj.MjData(model)

dt = model.opt.timestep
T = 5.0
steps = int(T / dt)

ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "end-effector")

# ==============================
# Эталонная траектория EE
# ==============================
t = np.linspace(0, T, steps)
xee_ref = np.zeros((steps, 3))

mj.mj_forward(model, data)
xee0 = data.xpos[ee_id].copy()

A = 0.05
xee_ref[:, 0] = xee0[0] + A * np.sin(2*np.pi*t/T)
xee_ref[:, 1] = xee0[1]
xee_ref[:, 2] = xee0[2] + A * np.cos(2*np.pi*t/T)

# ==============================
# Эталонная суставная траектория
# ==============================
q_ref = np.zeros((steps, model.nq))
dq_ref = np.zeros_like(q_ref)

q_ref[:] = data.qpos
dq_ref[:] = 0.0

# ==============================
# Симуляция эпизода (POSITION CONTROL)
# ==============================
def simulate_episode(Kp, Kd):
    mj.mj_resetData(model, data)
    loss = 0.0

    for i in range(steps):
        q = data.qpos.copy()
        dq = data.qvel.copy()

        # ВНЕШНИЙ PD → позиционная команда
        q_cmd = q + Kp * (q_ref[i] - q) + Kd * (dq_ref[i] - dq)

        # Ограничение (как у реального робота)
        q_cmd = np.clip(q_cmd,
                        model.jnt_range[:, 0],
                        model.jnt_range[:, 1])

        data.ctrl[:] = q_cmd

        mj.mj_step(model, data)

        ee_pos = data.xpos[ee_id]
        loss += np.sum((ee_pos - xee_ref[i])**2)

    return loss

# ==============================
# Оптимизация PD
# ==============================
Kp = np.ones(model.nq) * 0.5
Kd = np.ones(model.nq) * 0.05

lr = 5e-2
eps = 1e-3
n_iters = 25

for it in range(n_iters):
    base_loss = simulate_episode(Kp, Kd)

    grad_Kp = np.zeros_like(Kp)
    grad_Kd = np.zeros_like(Kd)

    for j in range(model.nq):
        Kp_eps = Kp.copy()
        Kp_eps[j] += eps
        grad_Kp[j] = (simulate_episode(Kp_eps, Kd) - base_loss) / eps

        Kd_eps = Kd.copy()
        Kd_eps[j] += eps
        grad_Kd[j] = (simulate_episode(Kp, Kd_eps) - base_loss) / eps

    Kp -= lr * grad_Kp
    Kd -= lr * grad_Kd

    print(f"Iter {it:02d} | Loss = {base_loss:.6f}")
    print("  Kp:", Kp)
    print("  Kd:", Kd)

print("\nOptimized gains:")
print("Kp =", Kp)
print("Kd =", Kd)
