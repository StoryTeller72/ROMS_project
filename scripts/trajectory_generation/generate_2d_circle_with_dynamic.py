import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from pathlib import Path

class AdaptivePDControllerForPositionActuators:
    """
    Адаптивный ПД-регулятор для position актуаторов
    
    Идея: вместо прямого управления моментами, корректируем 
    желаемую позицию на основе ошибки и нагрузки
    """
    def __init__(self, model, Kp_correction, Kd_correction, payload_mass=0.0):
        self.model = model
        self.nq = model.nq
        
        # Коэффициенты коррекции (не моменты, а поправки к углам!)
        self.Kp_correction_base = np.array(Kp_correction)
        self.Kd_correction_base = np.array(Kd_correction)
        
        self.Kp_correction = self.Kp_correction_base.copy()
        self.Kd_correction = self.Kd_correction_base.copy()
        
        # Интегральная составляющая для компенсации постоянных ошибок
        self.Ki_correction = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        self.integral_error = np.zeros(self.nq)
        self.integral_limit = np.array([0.1, 0.1, 0.08, 0.06, 0.04, 0.02])
        
        self.payload_mass = payload_mass
        self.update_gains_for_payload(payload_mass)
        
    def update_gains_for_payload(self, payload_mass):
        """
        Адаптация коэффициентов под нагрузку
        """
        self.payload_mass = payload_mass
        
        # Коэффициент адаптации
        adaptation_factor = 1.0 + payload_mass * 0.8
        ,
        # Веса для разных суставов (дистальные - сильнее)
        adaptation_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        self.Kp_correction = self.Kp_correction_base * (1.0 + (adaptation_factor - 1.0) * adaptation_weights)
        self.Kd_correction = self.Kd_correction_base * (1.0 + (adaptation_factor - 1.0) * adaptation_weights)
        
        print(f"\n=== Адаптация под нагрузку {payload_mass:.3f} кг ===")
        print(f"Kp_correction: {self.Kp_correction}")
        print(f"Kd_correction: {self.Kd_correction}")
    
    def compute_corrected_position(self, data, q_desired, dq_desired, dt):
        """
        Вычисление скорректированной желаемой позиции
        
        Вместо вычисления момента, вычисляем поправку к желаемому углу
        
        Returns:
            q_corrected: скорректированная желаемая позиция для актуаторов
            q_error: ошибка по углам
        """
        q_current = data.qpos.copy()
        dq_current = data.qvel.copy()
        
        # Ошибки
        q_error = q_desired - q_current
        dq_error = dq_desired - dq_current
        
        # Обновление интегральной составляющей
        self.integral_error += q_error * dt
        # Ограничение интеграла (anti-windup)
        self.integral_error = np.clip(self.integral_error, 
                                       -self.integral_limit, 
                                       self.integral_limit)
        
        # ПИД-поправка к желаемому углу
        position_correction = (
            self.Kp_correction * q_error + 
            self.Kd_correction * dq_error +
            self.Ki_correction * self.integral_error
        )
        
        # Ограничение коррекции (чтобы не уйти слишком далеко)
        max_correction = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])  # радианы
        position_correction = np.clip(position_correction, 
                                       -max_correction, 
                                       max_correction)
        
        # Скорректированная желаемая позиция
        q_corrected = q_desired + position_correction
        
        # Проверка на ограничения суставов
        for i in range(self.nq):
            q_corrected[i] = np.clip(q_corrected[i], 
                                      self.model.jnt_range[i, 0], 
                                      self.model.jnt_range[i, 1])
        
        return q_corrected, q_error, dq_error, position_correction
    
    def reset_integral(self):
        """Сброс интегральной составляющей"""
        self.integral_error = np.zeros(self.nq)


def get_circular_trajectory_with_adaptive_pd_position(trajectory_name, payload_mass=0.0):
    """
    Генерация траектории с адаптивным ПД для position актуаторов
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
    # НАСТРОЙКА АДАПТИВНОГО КОНТРОЛЛЕРА
    # ============================================
    
    # Коэффициенты коррекции позиции (не моменты!)
    # Эти значения определяют, насколько сильно корректировать желаемую позицию
    # при наличии ошибки. Меньшие значения = более консервативная коррекция
    Kp_correction = np.array([0.8, 0.8, 0.6, 0.5, 0.4, 0.3])
    Kd_correction = np.array([0.15, 0.15, 0.12, 0.1, 0.15, 0.15])
    
    controller = AdaptivePDControllerForPositionActuators(
        model, Kp_correction, Kd_correction, payload_mass
    )

    # ============================================
    # ГЕНЕРАЦИЯ ТРАЕКТОРИИ (аналогично предыдущему)
    # ============================================
    
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "end-effector")
    
    data.qpos[:] = np.array([0, 0, 0, 0, 0, 0])
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
            
        for iteration in range(max_iter):
            data.qpos[:] = q
            mj.mj_forward(model, data)
            ee_pos = data.xpos[ee_id]
            err = target - ee_pos
            
            if np.linalg.norm(err) < tol:
                break
            
            jacp = np.zeros((3, model.nv))
            mj.mj_jacBody(model, data, jacp, np.zeros_like(jacp), ee_id)
            
            lambda_damping = 0.01
            J_damped = jacp.T @ jacp + lambda_damping * np.eye(model.nv)
            dq = lr * np.linalg.solve(J_damped, jacp.T @ err)
            dq = np.clip(dq, -2.5, 2.5)
            q[:len(dq)] += dq
            
            for i in range(model.nq):
                q[i] = np.clip(q[i], model.jnt_range[i, 0], model.jnt_range[i, 1])
        
        return q

    joint_waypoints = np.zeros((n_points, model.nq))
    print("Вычисление IK...")
    q_prev = data.qpos.copy()
    
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
    # СИМУЛЯЦИЯ С АДАПТИВНЫМ ПД
    # ============================================
    
    data.qpos[:] = q_traj[0]
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)
    
    # Стабилизация
    print("Стабилизация...")
    controller.reset_integral()
    
    for _ in range(1000):
        q_corrected, _, _, _ = controller.compute_corrected_position(
            data, q_traj[0], np.zeros(model.nq), dt
        )
        data.ctrl[:] = q_corrected  # Position актуаторы ожидают углы
        mj.mj_step(model, data)
    
    print(f"Ошибка после стабилизации: {np.linalg.norm(data.qpos - q_traj[0]):.6f} рад")
    
    viewer = mj.viewer.launch_passive(model, data)
    
    # Массивы для данных
    ee_positions = []
    ee_positions_desired = []
    q_actual = []
    q_commanded = []  # Что мы послали в актуаторы
    tracking_errors_q = []
    tracking_errors_dq = []
    position_corrections = []
    
    print("Запуск траектории с адаптивным ПД...")
    
    for i in range(len(t_fine)):
        q_des = q_traj[i]
        dq_des = dq_traj[i]
        
        # Вычисление скорректированной позиции
        q_corrected, q_error, dq_error, correction = controller.compute_corrected_position(
            data, q_des, dq_des, dt
        )
        
        # Проверка на валидность
        if np.any(np.isnan(q_corrected)) or np.any(np.isinf(q_corrected)):
            print(f"ОШИБКА: NaN/Inf в q_corrected на шаге {i}")
            break
        
        # Задаем скорректированную позицию в актуаторы
        data.ctrl[:] = q_corrected
        
        # Шаг симуляции
        mj.mj_step(model, data)
        
        if np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos)):
            print(f"ОШИБКА: NaN/Inf в qpos на шаге {i}")
            break
        
        # Запись данных
        ee_positions.append(data.xpos[ee_id].copy())
        q_actual.append(data.qpos.copy())
        q_commanded.append(q_corrected.copy())
        
        tracking_errors_q.append(np.linalg.norm(q_error))
        tracking_errors_dq.append(np.linalg.norm(dq_error))
        position_corrections.append(correction.copy())
        
        # Желаемая позиция
        data_temp = mj.MjData(model)
        data_temp.qpos[:] = q_des
        mj.mj_forward(model, data_temp)
        ee_positions_desired.append(data_temp.xpos[ee_id].copy())
        
        if i % 250 == 0:
            print(f"Шаг {i}/{len(t_fine)}, ошибка: {tracking_errors_q[-1]:.4f} рад, "
                  f"коррекция: {np.linalg.norm(correction):.4f} рад")
        
        viewer.sync()

    viewer.close()
    print("Симуляция завершена")

    # Преобразование
    ee_positions = np.array(ee_positions)
    ee_positions_desired = np.array(ee_positions_desired)
    q_actual = np.array(q_actual)
    q_commanded = np.array(q_commanded)
    position_corrections = np.array(position_corrections)

    # ============================================
    # СОХРАНЕНИЕ
    # ============================================
    
    suffix = f"_payload_{payload_mass:.2f}kg_position_actuators"
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_endeffector.npy', ee_positions)
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_q_actual.npy', q_actual)
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_q_desired.npy', q_traj)
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_q_commanded.npy', q_commanded)
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_corrections.npy', position_corrections)
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_Kp.npy', controller.Kp_correction)
    np.save(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_Kd.npy', controller.Kd_correction)
    
    print(f"Данные сохранены в {save_path}")

    # ============================================
    # СТАТИСТИКА
    # ============================================
    
    ee_error = np.linalg.norm(ee_positions - ee_positions_desired, axis=1)
    
    print(f"\n{'='*60}")
    print(f"СТАТИСТИКА ДЛЯ НАГРУЗКИ {payload_mass} КГ (POSITION ACTUATORS)")
    print(f"{'='*60}")
    print(f"Позиция эндэффектора:")
    print(f"  Средняя ошибка: {np.mean(ee_error)*1000:.2f} мм")
    print(f"  Максимальная ошибка: {np.max(ee_error)*1000:.2f} мм")
    print(f"  RMS ошибка: {np.sqrt(np.mean(ee_error**2))*1000:.2f} мм")
    print(f"\nУглы суставов:")
    print(f"  Средняя ошибка: {np.mean(tracking_errors_q):.4f} рад ({np.rad2deg(np.mean(tracking_errors_q)):.2f}°)")
    print(f"  Максимальная ошибка: {np.max(tracking_errors_q):.4f} рад ({np.rad2deg(np.max(tracking_errors_q)):.2f}°)")
    print(f"\nКоррекция позиции:")
    print(f"  Средняя коррекция: {np.mean(np.linalg.norm(position_corrections, axis=1)):.4f} рад")
    print(f"  Максимальная коррекция: {np.max(np.linalg.norm(position_corrections, axis=1)):.4f} рад")
    print(f"  Коррекции по суставам (средние): {np.mean(np.abs(position_corrections), axis=0)}")
    print(f"{'='*60}\n")

    # ============================================
    # ГРАФИКИ
    # ============================================
    
    # 1. Углы - желаемые, командованные, фактические
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()
    for j in range(model.nq):
        axes[j].plot(t_fine, q_traj[:, j], 'b--', label='Desired', linewidth=2, alpha=0.7)
        axes[j].plot(t_fine, q_commanded[:, j], 'g:', label='Commanded', linewidth=2, alpha=0.7)
        axes[j].plot(t_fine, q_actual[:, j], 'r-', label='Actual', linewidth=1)
        axes[j].set_title(f'Joint {j+1} (Kp_corr={controller.Kp_correction[j]:.2f})')
        axes[j].set_xlabel('Time [s]')
        axes[j].set_ylabel('Angle [rad]')
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(fontsize=8)
    plt.suptitle(f'Joint Angles - Payload: {payload_mass} kg (Position Actuators)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_joints.png', dpi=150)
    plt.show()

    # 2. Коррекция позиции
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()
    for j in range(model.nq):
        axes[j].plot(t_fine, position_corrections[:, j] * 1000, 'purple', linewidth=1)
        axes[j].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[j].set_title(f'Joint {j+1} Position Correction')
        axes[j].set_xlabel('Time [s]')
        axes[j].set_ylabel('Correction [mrad]')
        axes[j].grid(True, alpha=0.3)
    plt.suptitle(f'Position Corrections - Payload: {payload_mass} kg', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_corrections.png', dpi=150)
    plt.show()

    # 3. Ошибки
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(t_fine, tracking_errors_q, 'r-', linewidth=1)
    ax1.set_title('Joint Position Tracking Error')
    ax1.set_ylabel('||q_error|| [rad]')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.01, color='orange', linestyle='--', label='0.01 rad threshold')
    ax1.legend()
    
    ax2.plot(t_fine, ee_error * 1000, 'm-', linewidth=1)
    ax2.set_title('End-Effector Position Error')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Error [mm]')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=10, color='orange', linestyle='--', label='10 mm threshold')
    ax2.legend()
    
    plt.suptitle(f'Tracking Errors - Payload: {payload_mass} kg', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_errors.png', dpi=150)
    plt.show()

    # 4. 3D траектория
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ee_positions_desired[:, 0], ee_positions_desired[:, 1], ee_positions_desired[:, 2], 
            'b--', label='Desired', linewidth=3, alpha=0.6)
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
            'r-', label='Actual', linewidth=2)
    ax.scatter([ee_positions[0, 0]], [ee_positions[0, 1]], [ee_positions[0, 2]], 
               c='green', s=150, marker='o', label='Start', edgecolors='black', linewidths=2)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_zlabel('Z [m]', fontsize=12)
    ax.set_title(f'End-Effector 3D Trajectory - Payload: {payload_mass} kg', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.savefig(save_path + f'/complicated_trajectory/{trajectory_name}{suffix}_3d.png', dpi=150)
    plt.show()
    
    return {
        'ee_error_mean': np.mean(ee_error) * 1000,
        'ee_error_max': np.max(ee_error) * 1000,
        'q_error_mean': np.mean(tracking_errors_q),
        'q_error_max': np.max(tracking_errors_q),
        'correction_mean': np.mean(np.linalg.norm(position_corrections, axis=1)),
        'correction_max': np.max(np.linalg.norm(position_corrections, axis=1)),
        'Kp': controller.Kp_correction,
        'Kd': controller.Kd_correction
    }


if __name__ == "__main__":
    # Тест с одной нагрузкой
    get_circular_trajectory_with_adaptive_pd_position("circle", payload_mass=0.0001)
    
    # Сравнение разных нагрузок
    # payloads = [0.0, 0.5, 1.0, 2.0]
    # for p in payloads:
    #     get_circular_trajectory_with_adaptive_pd_position("circle", payload_mass=p)