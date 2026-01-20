import mujoco as mj
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "robot" / "robot.xml"

model = mj.MjModel.from_xml_path(str(MODEL_PATH)) 
data = mj.MjData(model)

dt = model.opt.timestep


A = np.array([0.5, 0.3, 0.4, 0.2, 0.2, 0.1])   
w = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])   
phi = np.zeros(6)                            

q0 = np.zeros(6) 


with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0

    while viewer.is_running():
        q_cmd = q0 + A * np.sin(w * t + phi)
        data.ctrl[:] = q_cmd
        mj.mj_step(model, data)
        viewer.sync()
        t += dt
        time.sleep(dt)
