import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from pathlib import Path

def get_circular_trajectory_no_controller(trajectory_name, payload_mass=0.0):
    """
    Генерация траектории для position actuators без использования адаптивного ПД.
    Просто подаем желаемые позиции в actuators.
    """
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "robot" / "robot.xml"
    SAVE_PATH = PROJECT_ROOT / "data"
    save_path = str(SAVE_PATH)
    xml_path = str(MODEL_PATH)

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    
    # Установка массы нагрузки
    payload_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "payload")
    if payload_body_id >= 0:
        model.body_mass[payload_body_id] = payload_mass
        print(f"Установлена масса нагрузки: {payload_mass} кг")
    
    dt = model.opt.timestep
    t_total = 10.0

    # ============================================
    # ГЕНЕРАЦИЯ ТРАЕКТОРИИ
    # ============================================
    
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "end-effector")
    
    data.qpos[:] = np.zeros(model.nq)
    mj.mj_forward(model, data)
    ee_start = data.xpos[ee_id].copy()
    print(f"EE start position: {ee_start}")

    n_points = 2000
    t_waypoints = np.linspace(0, t_total, n_points)
    radius = 0.3

    circle_center = ee_start.copy()
    circle_center[2] -= radius

    theta = np.linspace(0, 2*np.pi, n_points)
    ee_traj = np.zeros((n_points, 3))
    ee_traj[:, 0] = circle_center[0] + radius * np.sin(theta)
    ee_traj[:, 1] = circle_center[1]
    ee_traj[:, 2] = circle_center[2] + radius * np.cos(theta)

    def inverse_kinematics(target, q_init=None, max_iter=200, tol=1e-3, lr=0.2):
        if q_init is None:
            q = data.qpos.copy()
        else:
            q = q_init.copy()
        for _ in range(max_iter):
            data.qpos[:] = q
            mj.mj_forward(model, data)
            err = target - data.xpos[ee_id]
            if np.linalg.norm(err) < tol:
                break
            jacp = np.zeros((3, model.nv))
            mj.mj_jacBody(model, data, jacp, np.zeros_like(jacp), ee_id)
            lambda_damping = 0.01
            J_damped = jacp.T @ jacp + lambda_damping * np.eye(model.nv)
            dq = lr * np.linalg.solve(J_damped, jacp.T @ err)
            dq = np.clip(dq, -2.5, 2.5)
            q[:len(dq)] += dq
            q[:model.nq] = np.clip(q[:model.nq], model.jnt_range[:, 0], model.jnt_range[:, 1])
        return q

    joint_waypoints = np.zeros((n_points, model.nq))
    q_prev = data.qpos.copy()
    print("Вычисление IK...")
    for i in range(n_points):
        joint_waypoints[i] = inverse_kinematics(ee_traj[i], q_init=q_prev)
        q_prev = joint_waypoints[i].copy()

    t_fine = np.arange(0, t_total, dt)
    q_traj = np.zeros((len(t_fine), model.nq))
    dq_traj = np.zeros_like(q_traj)
    
    for j in range(model.nq):
        cs = CubicSpline(t_waypoints, joint_waypoints[:, j], bc_type='natural')
        q_traj[:, j] = cs(t_fine)
        dq_traj[:, j] = cs(t_fine, 1)

    # ============================================
    # СИМУЛЯЦИЯ БЕЗ КОНТРОЛЛЕРА
    # ============================================
    
    data.qpos[:] = q_traj[0]
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)
    
    viewer = mj.viewer.launch_passive(model, data)
    
    ee_positions = []
    ee_positions_desired = []
    q_actual = []
    q_commanded = []

    print("Запуск траектории без контроллера...")
    for i in range(len(t_fine)):
        q_des = q_traj[i]
        dq_des = dq_traj[i]

        # Подавать желаемые позиции напрямую
        data.ctrl[:] = q_des
        mj.mj_step(model, data)

        # Сохраняем данные
        ee_positions.append(data.xpos[ee_id].copy())
        q_actual.append(data.qpos.copy())
        q_commanded.append(q_des.copy())

        # Желаемая позиция эндэффектора
        data_temp = mj.MjData(model)
        data_temp.qpos[:] = q_des
        mj.mj_forward(model, data_temp)
        ee_positions_desired.append(data_temp.xpos[ee_id].copy())

        if i % 250 == 0:
            err = np.linalg.norm(data.qpos - q_des)
            print(f"Step {i}/{len(t_fine)}, ||q_error||={err:.4f} rad")

        viewer.sync()

    viewer.close()
    print("Симуляция завершена")

    # ============================================
    # Преобразование и сохранение
    # ============================================
    
    ee_positions = np.array(ee_positions)
    ee_positions_desired = np.array(ee_positions_desired)
    q_actual = np.array(q_actual)
    q_commanded = np.array(q_commanded)

    suffix = f"_payload_{payload_mass:.2f}kg_no_controller"
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_endeffector.npy', ee_positions)
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_q_actual.npy', q_actual)
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_q_desired.npy', q_traj)
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_q_commanded.npy', q_commanded)

    # 3D траектория
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ee_positions_desired[:, 0], ee_positions_desired[:, 1], ee_positions_desired[:, 2], 
            'b--', label='Desired', linewidth=3, alpha=0.6)
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
            'r-', label='Actual', linewidth=2)
    ax.scatter([ee_positions[0, 0]], [ee_positions[0, 1]], [ee_positions[0, 2]], 
               c='green', s=150, marker='o', label='Start', edgecolors='black', linewidths=2)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'End-Effector 3D Trajectory - Payload: {payload_mass} kg (No Controller)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.show()

    return {
        'ee_positions': ee_positions,
        'ee_positions_desired': ee_positions_desired,
        'q_actual': q_actual,
        'q_commanded': q_commanded
    }

if __name__ == "__main__":
    get_circular_trajectory_no_controller("circle__test", payload_mass=0.5)
