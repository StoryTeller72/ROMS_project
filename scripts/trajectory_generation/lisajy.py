import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def get_lissajous_3d_trajectory(trajectory_name):
    # -------------------------------
    # Пути к файлам
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "robot" / "robot.xml"
    SAVE_PATH = PROJECT_ROOT / "data"
    save_path = str(SAVE_PATH) 
    xml_path = str(MODEL_PATH)

    # -------------------------------
    # Загружаем модель
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    dt = model.opt.timestep
    t_total = 10
    n_points = 7

    # -------------------------------
    # ID эндэффектора и начальная позиция
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "end-effector")
    mj.mj_forward(model, data)
    ee_start = data.xpos[ee_id].copy()
    ee_start[2] -= 0.3
    print("EE start position:", ee_start)

    # -------------------------------
    # Параметры кривой Лиссажу
    ax, ay, az = 0.3, 0.3 , 0.1       # амплитуды
    fx, fy, fz = 2.0, 2, 1.0          # частоты
    dx, dy, dz = 0, np.pi/2, np.pi/2    # фазы

    t_waypoints = np.linspace(0, t_total, n_points)
    ee_traj = np.zeros((n_points, 3))
    ee_traj[:, 0] = ee_start[0] + ax * np.sin(2*np.pi*fx*t_waypoints/t_total + dx)
    ee_traj[:, 1] = ee_start[1] + ay * np.sin(2*np.pi*fy*t_waypoints/t_total + dy)
    ee_traj[:, 2] = ee_start[2] + az * np.sin(2*np.pi*fz*t_waypoints/t_total + dz)

    # -------------------------------
    # Функция IK с damped pseudoinverse
    def inverse_kinematics_damped(target, q_init, model, data, ee_id, max_iter=200, tol=1e-4, damping=0.01, dq_max=0.1):
        q = q_init.copy()
        nq = len(q)
        for _ in range(max_iter):
            data.qpos[:] = q
            mj.mj_forward(model, data)
            ee_pos = data.xpos[ee_id]
            err = target - ee_pos
            if np.linalg.norm(err) < tol:
                break
            # Якоби
            J = np.zeros((3, nq))
            mj.mj_jacBody(model, data, J, None, ee_id)
            # Damped pseudoinverse
            JTJ = J @ J.T + (damping**2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JTJ, err)
            # Ограничение максимальной скорости
            dq = np.clip(dq, -dq_max, dq_max)
            q += dq
        return q

    # -------------------------------
    # Генерируем опорные точки для суставов
    joint_waypoints = np.zeros((n_points, model.nq))
    q_init = data.qpos.copy()
    for i in range(n_points):
        joint_waypoints[i] = inverse_kinematics_damped(ee_traj[i], q_init, model, data, ee_id)
        q_init = joint_waypoints[i]  # плавность

    # -------------------------------
    # Плавная траектория через CubicSpline
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

    # -------------------------------
    # Сохраняем координаты эндэффектора
    ee_positions = np.array(ee_positions)
    np.save(save_path + f'/lissajous_3d_ee_posExample.npy', ee_positions)
    np.save(save_path + f'/lissajous_3d_qExample.npy', q_traj)
    np.save(save_path + f'/lissajous_3d_dqExample.npy', dq_traj)
    print(f"Координаты эндэффектора сохранены в {save_path}")

    # -------------------------------
    # Графики
    plt.figure(figsize=(12,6))
    for j in range(model.nq):
        plt.plot(t_fine, q_traj[:, j], label=f'Joint {j+1}')
    plt.title("Joint Angles")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(t_fine, ee_positions[:, 0], label='X')
    plt.plot(t_fine, ee_positions[:, 1], label='Y')
    plt.plot(t_fine, ee_positions[:, 2], label='Z')
    plt.title("End-Effector Global Position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.grid(True)
    plt.legend()
    plt.show()


   

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Вся траектория (сплайн) — зелёная линия
    ax.plot(
        ee_positions[:, 0],
        ee_positions[:, 1],
        ee_positions[:, 2],
        color='green',
        linewidth=2,
        label='Spline trajectory'
    )

    # Опорные точки — красные маркеры
    ax.scatter(
        ee_traj[:, 0],
        ee_traj[:, 1],
        ee_traj[:, 2],
        color='red',
        s=60,
        label='Waypoints'
    )

    ax.set_title("Lissajous trajectory of end-effector")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.grid(True)

    plt.show()


if __name__ == "__main__":
    get_lissajous_3d_trajectory("lissajous3d")
