import mujoco as mj
import mujoco.viewer
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "robot" / "robot.xml"

class EndEffectorPathGenerator:
    def __init__(self, model_path):
        self.model = mj.MjModel.from_xml_path(str(model_path))
        self.data = mj.MjData(self.model)
        self.dt = self.model.opt.timestep
        
        # Найти индекс end effector'а
        self.ee_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "end-effector")
        
        # Для хранения траектории
        self.ee_positions = []
        self.time_steps = []
        self.viewer = None
    
    def inverse_kinematics(self, target_pos, max_iterations=100, tolerance=1e-4):
        """
        Использует численный метод для вычисления углов суставов для достижения целевой позиции
        """
        def objective(q):
            # Копируем текущее состояние
            data_temp = mj.MjData(self.model)
            data_temp.qpos[:] = q
            mj.mj_kinematics(self.model, data_temp)
            
            # Получаем текущую позицию end effector'а
            ee_pos = data_temp.body(self.ee_id).xpos
            
            # Возвращаем ошибку позиции
            error = ee_pos - target_pos
            return error
        
        # Начальное приближение - текущие углы
        q_init = self.data.qpos.copy()
        
        # Используем least squares для решения IK
        result = least_squares(
            objective,
            q_init,
            bounds=(self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]),
            max_nfev=max_iterations
        )
        
        return result.x
    
    def jacobian_ik(self, target_pos, current_q, step_size=0.01, iterations=20):
        """
        Jacobian-based inverse kinematics метод (более быстрый)
        """
        q = current_q.copy()
        
        for _ in range(iterations):
            # Вычисляем текущую позицию end effector'а
            data_temp = mj.MjData(self.model)
            data_temp.qpos[:] = q
            mj.mj_kinematics(self.model, data_temp)
            ee_pos = data_temp.body(self.ee_id).xpos.copy()
            
            # Ошибка позиции
            error = target_pos - ee_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < 1e-4:
                break
            
            # Численное вычисление Jacobian
            delta = 1e-6
            J = np.zeros((3, len(q)))
            
            for i in range(len(q)):
                q_plus = q.copy()
                q_plus[i] += delta
                
                data_temp = mj.MjData(self.model)
                data_temp.qpos[:] = q_plus
                mj.mj_kinematics(self.model, data_temp)
                ee_pos_plus = data_temp.body(self.ee_id).xpos.copy()
                
                J[:, i] = (ee_pos_plus - ee_pos) / delta
            
            # Псевдообратная матрица Jacobian
            try:
                J_pinv = np.linalg.pinv(J)
                dq = J_pinv @ error
                q += step_size * dq
                
                # Ограничиваем углы
                q = np.clip(q, self.model.jnt_range[:, 0], self.model.jnt_range[:, 1])
            except:
                break
        
        return q
    
    def _draw_path_in_viewer(self, viewer):
        """Рисует траекторию в MuJoCo viewer"""
        if len(self.ee_positions) < 2:
            return
        
        positions = np.array(self.ee_positions)
        
        # Рисуем линию траектории (каждую 10-ю точку для оптимизации)
        for i in range(0, len(positions) - 1, 10):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 3:
                break
            
            start = positions[i]
            end = positions[i + 1]
            
            # Вычисляем параметры цилиндра (линии)
            diff = end - start
            length = np.linalg.norm(diff)
            
            if length < 1e-6:
                continue
            
            # Центр цилиндра
            center = (start + end) / 2
            
            # Размер: [радиус, радиус, половина длины]
            size = np.array([0.001, 0.001, length / 2], dtype=np.float64).reshape(3, 1)
            center_reshaped = center.astype(np.float64).reshape(3, 1)
            
            # Матрица ориентации (identity as 9-element vector)
            mat = np.eye(3, dtype=np.float64).flatten().reshape(9, 1)
            
            rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32).reshape(4, 1)
            
            mj.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mj.mjtGeom.mjGEOM_CYLINDER,
                size,
                center_reshaped,
                mat,
                rgba
            )
            viewer.user_scn.ngeom += 1
        
        # Рисуем начальную точку (зелёная сфера)
        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
            size = np.array([0.01, 0.01, 0.01], dtype=np.float64).reshape(3, 1)
            pos = positions[0].astype(np.float64).reshape(3, 1)
            mat = np.eye(3, dtype=np.float64).flatten().reshape(9, 1)
            rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32).reshape(4, 1)
            
            mj.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mj.mjtGeom.mjGEOM_SPHERE,
                size,
                pos,
                mat,
                rgba
            )
            viewer.user_scn.ngeom += 1
        
        # Рисуем конечную точку (красная сфера)
        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
            size = np.array([0.01, 0.01, 0.01], dtype=np.float64).reshape(3, 1)
            pos = positions[-1].astype(np.float64).reshape(3, 1)
            mat = np.eye(3, dtype=np.float64).flatten().reshape(9, 1)
            rgba = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(4, 1)
            
            mj.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mj.mjtGeom.mjGEOM_SPHERE,
                size,
                pos,
                mat,
                rgba
            )
            viewer.user_scn.ngeom += 1
    
    def generate_circular_path(self, center, radius, height_offset, duration=5.0):
        """Генерирует круговую траекторию в 3D пространстве с использованием IK"""
        self.ee_positions = []
        self.time_steps = []
        
        # Первый проход - собираем данные о траектории
        print("Собираю данные траектории с использованием IK...")
        temp_data = mj.MjData(self.model)
        temp_data.qpos[:] = self.data.qpos.copy()
        
        t = 0
        frame = 0
        while t < duration:
            # Целевая позиция - круг в 3D
            angle = (2 * np.pi * t) / duration
            target_x = center[0] + radius * np.cos(angle)
            target_y = center[1] + radius * np.sin(angle)
            target_z = center[2] + 0.1 * np.sin(2 * angle)  # Небольшое движение по Z
            target_pos = np.array([target_x, target_y, target_z])
            
            # Вычисляем требуемые углы суставов
            q_target = self.jacobian_ik(target_pos, temp_data.qpos.copy(), step_size=0.05, iterations=15)
            temp_data.qpos[:] = q_target
            
            mj.mj_kinematics(self.model, temp_data)
            
            # Получить позицию end effector'а
            ee_pos = temp_data.body(self.ee_id).xpos.copy()
            self.ee_positions.append(ee_pos)
            self.time_steps.append(t)
            
            t += self.dt * 5  # Ускоряем сбор данных
            frame += 1
            
            if frame % 50 == 0:
                print(f"  Время: {t:.2f}s, EE позиция: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
        
        # Второй проход - визуализация
        print("Визуализирую траекторию в MuJoCo...")
        self.data.qpos[:] = temp_data.qpos.copy()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -45
            viewer.cam.distance = 2.5
            
            t = 0
            frame = 0
            while viewer.is_running() and t < duration:
                angle = (2 * np.pi * t) / duration
                target_x = center[0] + radius * np.cos(angle)
                target_y = center[1] + radius * np.sin(angle)
                target_z = center[2] + 0.1 * np.sin(2 * angle)
                target_pos = np.array([target_x, target_y, target_z])
                
                # Вычисляем и применяем IK
                q_target = self.jacobian_ik(target_pos, self.data.qpos.copy(), step_size=0.05, iterations=10)
                self.data.qpos[:] = q_target
                
                mj.mj_step(self.model, self.data)
                
                if frame % 5 == 0:
                    viewer.user_scn.ngeom = 0
                    self._draw_path_in_viewer(viewer)
                
                viewer.sync()
                t += self.dt * 5
                frame += 1
        
        return np.array(self.ee_positions), np.array(self.time_steps)
    
    def generate_linear_path(self, start_pos, end_pos, duration=5.0):
        """Генерирует линейную траекторию с использованием IK"""
        self.ee_positions = []
        self.time_steps = []
        
        # Первый проход - собираем данные
        print("Собираю данные траектории с использованием IK...")
        temp_data = mj.MjData(self.model)
        temp_data.qpos[:] = self.data.qpos.copy()
        
        t = 0
        frame = 0
        while t < duration:
            alpha = t / duration
            target_pos = start_pos + alpha * (end_pos - start_pos)
            
            q_target = self.jacobian_ik(target_pos, temp_data.qpos.copy(), step_size=0.05, iterations=15)
            temp_data.qpos[:] = q_target
            
            mj.mj_kinematics(self.model, temp_data)
            
            ee_pos = temp_data.body(self.ee_id).xpos.copy()
            self.ee_positions.append(ee_pos)
            self.time_steps.append(t)
            
            t += self.dt * 5
            frame += 1
            
            if frame % 30 == 0:
                print(f"  Прогресс: {alpha*100:.1f}%, EE позиция: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
        
        # Второй проход - визуализация
        print("Визуализирую траекторию в MuJoCo...")
        self.data.qpos[:] = temp_data.qpos.copy()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -45
            viewer.cam.distance = 2.5
            
            t = 0
            frame = 0
            while viewer.is_running() and t < duration:
                alpha = t / duration
                target_pos = start_pos + alpha * (end_pos - start_pos)
                
                q_target = self.jacobian_ik(target_pos, self.data.qpos.copy(), step_size=0.05, iterations=10)
                self.data.qpos[:] = q_target
                
                mj.mj_step(self.model, self.data)
                
                if frame % 5 == 0:
                    viewer.user_scn.ngeom = 0
                    self._draw_path_in_viewer(viewer)
                
                viewer.sync()
                t += self.dt * 5
                frame += 1
        
        return np.array(self.ee_positions), np.array(self.time_steps)
    
    def generate_lissajous_path(self, center, duration=5.0, freq_x=2, freq_y=3, amp_x=0.2, amp_y=0.2):
        """Генерирует кривую Лиссажу с использованием IK"""
        self.ee_positions = []
        self.time_steps = []
        
        # Первый проход - собираем данные
        print("Собираю данные траектории с использованием IK...")
        temp_data = mj.MjData(self.model)
        temp_data.qpos[:] = self.data.qpos.copy()
        
        t = 0
        frame = 0
        while t < duration:
            angle = (2 * np.pi * t) / duration
            x_offset = amp_x * np.sin(freq_x * angle)
            y_offset = amp_y * np.sin(freq_y * angle)
            
            target_pos = center + np.array([x_offset, y_offset, 0])
            
            q_target = self.jacobian_ik(target_pos, temp_data.qpos.copy(), step_size=0.05, iterations=15)
            temp_data.qpos[:] = q_target
            
            mj.mj_kinematics(self.model, temp_data)
            
            ee_pos = temp_data.body(self.ee_id).xpos.copy()
            self.ee_positions.append(ee_pos)
            self.time_steps.append(t)
            
            t += self.dt * 5
            frame += 1
        
        # Второй проход - визуализация
        print("Визуализирую траекторию в MuJoCo...")
        self.data.qpos[:] = temp_data.qpos.copy()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -45
            viewer.cam.distance = 2.5
            
            t = 0
            frame = 0
            while viewer.is_running() and t < duration:
                angle = (2 * np.pi * t) / duration
                x_offset = amp_x * np.sin(freq_x * angle)
                y_offset = amp_y * np.sin(freq_y * angle)
                
                target_pos = center + np.array([x_offset, y_offset, 0])
                
                q_target = self.jacobian_ik(target_pos, self.data.qpos.copy(), step_size=0.05, iterations=10)
                self.data.qpos[:] = q_target
                
                mj.mj_step(self.model, self.data)
                
                if frame % 5 == 0:
                    viewer.user_scn.ngeom = 0
                    self._draw_path_in_viewer(viewer)
                
                viewer.sync()
                t += self.dt * 5
                frame += 1
        
        return np.array(self.ee_positions), np.array(self.time_steps)
    
    def save_path_to_csv(self, filename):
        """Сохраняет траекторию в CSV файл"""
        import csv
        
        positions = np.array(self.ee_positions)
        time_steps = np.array(self.time_steps)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time (s)', 'X (m)', 'Y (m)', 'Z (m)'])
            for t, pos in zip(time_steps, positions):
                writer.writerow([t, pos[0], pos[1], pos[2]])
        
        print(f"Траектория сохранена в {filename}")
    
    def _draw_path_in_viewer(self, viewer):
        """Рисует траекторию в MuJoCo viewer"""
        if len(self.ee_positions) < 2:
            return
        
        positions = np.array(self.ee_positions)
        
        # Рисуем линию траектории (каждую 10-ю точку для оптимизации)
        for i in range(0, len(positions) - 1, 10):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 3:
                break
            
            start = positions[i]
            end = positions[i + 1]
            
            # Вычисляем параметры цилиндра (линии)
            diff = end - start
            length = np.linalg.norm(diff)
            
            if length < 1e-6:
                continue
            
            # Центр цилиндра
            center = (start + end) / 2
            
            # Размер: [радиус, радиус, половина длины]
            size = np.array([0.001, 0.001, length / 2], dtype=np.float64).reshape(3, 1)
            center_reshaped = center.astype(np.float64).reshape(3, 1)
            
            # Матрица ориентации (identity as 9-element vector)
            mat = np.eye(3, dtype=np.float64).flatten().reshape(9, 1)
            
            rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32).reshape(4, 1)
            
            mj.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mj.mjtGeom.mjGEOM_CYLINDER,
                size,
                center_reshaped,
                mat,
                rgba
            )
            viewer.user_scn.ngeom += 1
        
        # Рисуем начальную точку (зелёная сфера)
        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
            size = np.array([0.01, 0.01, 0.01], dtype=np.float64).reshape(3, 1)
            pos = positions[0].astype(np.float64).reshape(3, 1)
            mat = np.eye(3, dtype=np.float64).flatten().reshape(9, 1)
            rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32).reshape(4, 1)
            
            mj.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mj.mjtGeom.mjGEOM_SPHERE,
                size,
                pos,
                mat,
                rgba
            )
            viewer.user_scn.ngeom += 1
        
        # Рисуем конечную точку (красная сфера)
        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
            size = np.array([0.01, 0.01, 0.01], dtype=np.float64).reshape(3, 1)
            pos = positions[-1].astype(np.float64).reshape(3, 1)
            mat = np.eye(3, dtype=np.float64).flatten().reshape(9, 1)
            rgba = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(4, 1)
            
            mj.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mj.mjtGeom.mjGEOM_SPHERE,
                size,
                pos,
                mat,
                rgba
            )
            viewer.user_scn.ngeom += 1
    
    def generate_circular_path(self, center, radius, height_offset, duration=5.0):
        """Генерирует круговую траекторию в плоскости XY с визуализацией в MuJoCo"""
        self.ee_positions = []
        self.time_steps = []
        
        t = 0
        is_first_loop = True
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -45
            viewer.cam.distance = 2.5
            
            # Первый проход - собираем данные о траектории
            print("Собираю данные траектории...")
            temp_data = mj.MjData(self.model)
            t = 0
            while t < duration:
                angle = (2 * np.pi * t) / duration
                
                # Управление суставами для круговой траектории
                temp_data.ctrl[:] = np.array([0.5, 0.3, 0.4, 0.2, 0.2, 0.1]) * np.sin(2 * np.pi * t / duration + np.arange(6) * np.pi/3)
                
                mj.mj_step(self.model, temp_data)
                
                # Получить позицию end effector'а
                ee_pos = temp_data.body(self.ee_id).xpos.copy()
                self.ee_positions.append(ee_pos)
                self.time_steps.append(t)
                
                t += self.dt
            
            # Второй проход - визуализация с рисованием траектории
            print("Визуализирую траекторию в MuJoCo...")
            t = 0
            frame = 0
            while viewer.is_running() and t < duration:
                angle = (2 * np.pi * t) / duration
                
                # Управление суставами
                self.data.ctrl[:] = np.array([0.5, 0.3, 0.4, 0.2, 0.2, 0.1]) * np.sin(2 * np.pi * t / duration + np.arange(6) * np.pi/3)
                
                mj.mj_step(self.model, self.data)
                
                # Рисуем траекторию каждый N-й кадр
                if frame % 5 == 0:
                    viewer.user_scn.ngeom = 0
                    self._draw_path_in_viewer(viewer)
                
                viewer.sync()
                t += self.dt
                frame += 1
        
        return np.array(self.ee_positions), np.array(self.time_steps)
    
    def generate_linear_path(self, start_pos, end_pos, duration=5.0):
        """Генерирует линейную траекторию с визуализацией в MuJoCo"""
        self.ee_positions = []
        self.time_steps = []
        
        # Первый проход - собираем данные
        print("Собираю данные траектории...")
        temp_data = mj.MjData(self.model)
        t = 0
        while t < duration:
            alpha = t / duration
            
            temp_data.ctrl[:] = np.array([0.3, 0.2, 0.25, 0.15, 0.15, 0.05]) * np.sin(2 * np.pi * alpha)
            
            mj.mj_step(self.model, temp_data)
            
            ee_pos = temp_data.body(self.ee_id).xpos.copy()
            self.ee_positions.append(ee_pos)
            self.time_steps.append(t)
            
            t += self.dt
        
        # Второй проход - визуализация
        print("Визуализирую траекторию в MuJoCo...")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -45
            viewer.cam.distance = 2.5
            
            t = 0
            frame = 0
            while viewer.is_running() and t < duration:
                alpha = t / duration
                
                self.data.ctrl[:] = np.array([0.3, 0.2, 0.25, 0.15, 0.15, 0.05]) * np.sin(2 * np.pi * alpha)
                
                mj.mj_step(self.model, self.data)
                
                if frame % 5 == 0:
                    viewer.user_scn.ngeom = 0
                    self._draw_path_in_viewer(viewer)
                
                viewer.sync()
                t += self.dt
                frame += 1
        
        return np.array(self.ee_positions), np.array(self.time_steps)
    
    def generate_lissajous_path(self, duration=5.0, freq_x=2, freq_y=3, amp_x=0.2, amp_y=0.2):
        """Генерирует кривую Лиссажу с визуализацией в MuJoCo"""
        self.ee_positions = []
        self.time_steps = []
        
        # Первый проход - собираем данные
        print("Собираю данные траектории...")
        temp_data = mj.MjData(self.model)
        t = 0
        while t < duration:
            angle = (2 * np.pi * t) / duration
            
            temp_data.ctrl[:] = np.array([0.5, 0.3, 0.4, 0.2, 0.2, 0.1]) * np.sin(angle + np.arange(6) * np.pi/3)
            
            mj.mj_step(self.model, temp_data)
            
            ee_pos = temp_data.body(self.ee_id).xpos.copy()
            self.ee_positions.append(ee_pos)
            self.time_steps.append(t)
            
            t += self.dt
        
        # Второй проход - визуализация
        print("Визуализирую траекторию в MuJoCo...")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -45
            viewer.cam.distance = 2.5
            
            t = 0
            frame = 0
            while viewer.is_running() and t < duration:
                angle = (2 * np.pi * t) / duration
                
                self.data.ctrl[:] = np.array([0.5, 0.3, 0.4, 0.2, 0.2, 0.1]) * np.sin(angle + np.arange(6) * np.pi/3)
                
                mj.mj_step(self.model, self.data)
                
                if frame % 5 == 0:
                    viewer.user_scn.ngeom = 0
                    self._draw_path_in_viewer(viewer)
                
                viewer.sync()
                t += self.dt
                frame += 1
        
        return np.array(self.ee_positions), np.array(self.time_steps)
    
    def save_path_to_csv(self, filename):
        """Сохраняет траекторию в CSV файл"""
        import csv
        
        positions = np.array(self.ee_positions)
        time_steps = np.array(self.time_steps)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time (s)', 'X (m)', 'Y (m)', 'Z (m)'])
            for t, pos in zip(time_steps, positions):
                writer.writerow([t, pos[0], pos[1], pos[2]])
        
        print(f"Траектория сохранена в {filename}")


def main():
    # Создаем генератор траекторий
    generator = EndEffectorPathGenerator(str(MODEL_PATH))
    
    print("Выбери тип траектории:")
    print("1. Круговая траектория (полная 3D окружность)")
    print("2. Линейная траектория")
    print("3. Кривая Лиссажу")
    
    choice = input("Введи номер (1-3): ").strip()
    
    if choice == "1":
        print("\nГенерирую круговую траекторию с IK...")
        generator.generate_circular_path(
            center=np.array([0.5, 0.0, 0.8]),
            radius=0.25,
            height_offset=0.0,
            duration=8.0
        )
        
    elif choice == "2":
        print("\nГенерирую линейную траекторию с IK...")
        start = np.array([0.4, -0.2, 0.7])
        end = np.array([0.6, 0.2, 0.9])
        generator.generate_linear_path(start, end, duration=8.0)
        
    elif choice == "3":
        print("\nГенерирую кривую Лиссажу с IK...")
        center = np.array([0.5, 0.0, 0.8])
        generator.generate_lissajous_path(duration=8.0, freq_x=2, freq_y=3, amp_x=0.15, amp_y=0.15)
    
    # Сохранение траектории
    save = input("\nСохранить траекторию в CSV? (y/n): ").strip().lower()
    if save == 'y':
        output_path = PROJECT_ROOT / "data" / f"{choice}_trajectory.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generator.save_path_to_csv(str(output_path))


if __name__ == "__main__":
    main()
