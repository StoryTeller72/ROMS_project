import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.interpolate import CubicSpline
from pathlib import Path

def visualize_trajectory(positions, xml_path="/home/rustam/ROMS/models/robot/robot1joint.xml"):
    """
    Визуализирует траекторию эндэффектора в MuJoCo Viewer.
    
    positions: np.array, shape (N, 3) - глобальные позиции эндэффектора
    xml_path: путь к XML модели робота
    """
    # Загружаем модель
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    
    # Viewer
    viewer = mj.viewer.launch_passive(model, data)

    # Рисуем точки траектории
    for pos in positions:
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break  # достигнут лимит геомов
        
        size = np.array([0.001, 0.001, 0.001], dtype=np.float64).reshape(3,1)  # радиус сферы
        pos_reshaped = pos.astype(np.float64).reshape(3,1)
        mat = np.eye(3, dtype=np.float64).flatten().reshape(9,1)
        rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32).reshape(4,1)
        
        mj.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mj.mjtGeom.mjGEOM_SPHERE,
            size,
            pos_reshaped,
            mat,
            rgba
        )
        viewer.user_scn.ngeom += 1

    # Оставляем Viewer открытым
    while True:
        viewer.sync()

if __name__ == '__main__':
    positions = np.load('/home/rustam/ROMS/ee_positions.npy')
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "robot" / "robot.xml"
    visualize_trajectory(positions)