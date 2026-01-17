import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from pathlib import Path

def get_circular_trajectory(trajectory_name):
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "robot" / "robot.xml"
    SAVE_PATH = PROJECT_ROOT / "data"
    save_path = str(SAVE_PATH)
    xml_path = str(MODEL_PATH)

    # -------------------------------
    # Параметры симуляции
    # -------------------------------
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    dt = model.opt.timestep
    t_total = 5.0

    # -------------------------------
    # ID эндэффектора и начальная позиция
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "end-effector")
    mj.mj_forward(model, data)
    ee_start = data.xpos[ee_id].copy()  # верхняя точка круга
    print("EE start position:", ee_start)

    # -------------------------------
    # Создаём точки окружности в плоскости XZ
    n_points = 2000
    t_waypoints = np.linspace(0, t_total, n_points)
    radius = 0.3

    # Центр круга смещён вниз по Z на радиус, чтобы верхняя точка = ee_start
    circle_center = ee_start.copy()
    circle_center[2] -= radius

    theta = np.linspace(0, 2*np.pi, n_points)
    ee_traj = np.zeros((n_points, 3))
    ee_traj[:, 0] = circle_center[0] + radius * np.sin(theta)  # X
    ee_traj[:, 1] = circle_center[1]                         # Y (фиксируем)
    ee_traj[:, 2] = circle_center[2] + radius * np.cos(theta)  # Z
    # # -------------------------------
    # Простая ИК для каждого шага
    def inverse_kinematics(target, max_iter=500, tol=1e-4, lr=0.5):
        q = data.qpos.copy()
        for _ in range(max_iter):
            data.qpos[:] = q
            mj.mj_forward(model, data)
            ee_pos = data.xpos[ee_id]
            err = target - ee_pos
            if np.linalg.norm(err) < tol:
                break

            # Вычисляем Якоби (mj_jacBody)
            jacp = np.zeros((3, model.nv))
            mj.mj_jacBody(model, data, jacp, np.zeros_like(jacp), ee_id)

            # Псевдообратная матрица Якоби
            dq = lr * np.linalg.pinv(jacp) @ err
            q[:len(dq)] += dq  # только n DOF
        return q

    joint_waypoints = np.zeros((n_points, model.nq))
    for i in range(n_points):
        joint_waypoints[i] = inverse_kinematics(ee_traj[i])
    # -------------------------------
    # Генерация гладкой траектории через CubicSpline
    t_fine = np.arange(0, t_total, dt)
    q_traj = np.zeros((len(t_fine), model.nq))
    dq_traj = np.zeros_like(q_traj)
    for j in range(model.nq):
        cs = CubicSpline(t_waypoints, joint_waypoints[:, j], bc_type='clamped')
        q_traj[:, j] = cs(t_fine)
        dq_traj[:, j] = cs(t_fine, 1)

    # -------------------------------
    # Viewer и визуализация
    data.qpos[:] = q_traj[0]
    mj.mj_forward(model, data)
    viewer = mj.viewer.launch_passive(model, data)
    ee_positions = []

    for i in range(len(t_fine)):
        data.qpos[:] = q_traj[i]
        mj.mj_forward(model, data)
        ee_positions.append(data.xpos[ee_id].copy())
        viewer.sync()

    viewer.close()

    # # -------------------------------
    # Сохраняем координаты эндэффектора
    ee_positions = np.array(ee_positions)
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}_endeffector.npy', ee_positions)
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}_q.npy', q_traj)
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}_dq.npy', dq_traj)
    print(f"Координаты эндэффектора сохранены в {save_path}")

    # -------------------------------
    # # Графики
    plt.figure(figsize=(12, 6))
    for j in range(model.nq):
        plt.plot(t_fine, q_traj[:, j], label=f'Joint {j+1}')
    plt.title("Joint Angles")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(t_fine, ee_positions[:, 0], label='X')
    plt.plot(t_fine, ee_positions[:, 1], label='Y')
    plt.plot(t_fine, ee_positions[:, 2], label='Z')
    plt.title("End-Effector Global Position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    get_circular_trajectory("circle")
