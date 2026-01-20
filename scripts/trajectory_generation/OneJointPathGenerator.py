import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from pathlib import Path

def get_trajectory(joint_number):
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "robot"
    SAVE_PATH = PROJECT_ROOT / "data" 
    save_path = str(SAVE_PATH) +f'/link{joint_number}/link{joint_number}.npy'
    xml_path = str(MODEL_PATH) + f'/robot{joint_number}joint.xml'
    # -------------------------------
    # Параметры симуляции
    # -------------------------------
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    t_total = 3.0  # увеличиваем время для остановки
    dt = model.opt.timestep
    t = np.linspace(0, t_total, int(t_total/dt)*10)

    # -------------------------------
    # Опорная траектория: 0 -> π -> π (удержание) -> -π

    # Временные точки и значения
    # 0s -> 0
    # 1s -> π
    # 1.5s -> π (удержание)
    # 3s -> -π
    t_waypoints = np.array([0.0, 1.0, 1.5, 2.0, t_total])
    q_waypoints = np.array([0.0, np.pi / 2, 0, -np.pi/2, 0])

    # Кубический сплайн
    cs = CubicSpline(t_waypoints, q_waypoints, bc_type='clamped')

    # -------------------------------
    # Генерация позиции и скорости
    q_traj = cs(t)
    dq_traj = cs(t, 1)

    # -------------------------------
    # Начальное состояние
    data.qpos[:] = q_traj[0]
    mujoco.mj_forward(model, data)

    # -------------------------------
    # Viewer для визуализации
    viewer = mujoco.viewer.launch_passive(model, data)

    # Для записи данных
    q_record = []
    dq_record = []
    ee_positions = []

    # ID эндэффектора
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end-effector")

    # -------------------------------
    # Исполнение идеальной траектории
    for i in range(len(t)):
        data.qpos[0] = q_traj[i]  # двигаем только один сустав
        mujoco.mj_forward(model, data)  # без mj_step для идеального движения

        # Сохраняем данные
        q_record.append(data.qpos[0])
        dq_record.append(dq_traj[i])
        ee_positions.append(data.xpos[ee_id].copy())

        viewer.sync()

    viewer.close()

    # -------------------------------
    # Визуализация
    q_record = np.array(q_record)
    dq_record = np.array(dq_record)
    ee_positions = np.array(ee_positions)
    

    plt.figure(figsize=(12,6))

    plt.subplot(3,1,1)
    plt.plot(t, q_record, label='Joint Angle')
    plt.title("Joint Angle: 0 -> π -> π (hold) -> -π")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(t, dq_record, label='Joint Velocity', color='orange')
    plt.title("Joint Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [rad/s]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(t, ee_positions[:,0], label='X')
    plt.plot(t, ee_positions[:,1], label='Y')
    plt.plot(t, ee_positions[:,2], label='Z')
    plt.title("End-Effector Global Position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # -------------------------------
    # Сохраняем координаты эндэффектора
    np.save(save_path, ee_positions)
    print(f"Координаты эндэффектора сохранены в {save_path}")
    print(ee_positions[0])

if __name__ == '__main__':
    get_trajectory(1)