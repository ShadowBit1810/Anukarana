import os
import sys

# 1. Get the absolute path of Sim.py (F:\ZERO\Main\Sim.py)
file_path = os.path.abspath(__file__)

# 2. Go up 2 levels to reach the folder ABOVE 'ZERO'
# Level 1: F:\ZERO\Main
# Level 2: F:\ (This is where 'ZERO' lives)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

# 3. Add that root to sys.path so 'import ZERO' works
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4. Now this absolute import will always work
import taichi as ti
import ZERO.Main.config as config
from ZERO.Rendering import ui_backend as ui
from ZERO.Rendering import startup_tkinter
from ZERO.Operation import gravity_ops

# Initialize Taichi
config.initialise_taichi() # or ti.cpu depending on your setup

# 1. Run your initial tkinter screen (if it blocks until closed)
startup_tkinter.Screen()

# 2. Initialize the Taichi UI Manager
ui_manager = ui.UIManager()

gravity_solver = None  # created on first frame of STATE_SIM

# 3. The Main Render/Simulation Loop
while ui_manager.Win.window.running:
    ui_manager.draw()

    if config.CURRENT_STAGE == config.STATE_SIM:
        # Create solver once on first frame of sim
        if gravity_solver is None and "gravity" in ui_manager.ps.features:
            gravity_solver = gravity_ops.gravity(ui_manager.ps)

        if gravity_solver is not None:
            scaled_dt = config.DT * config.TIME_SCALE
            gravity_solver.compute_gravity(scaled_dt)

    else:
        # Reset solver if user goes back to sandbox
        gravity_solver = None

    ui_manager.Win.render_screen(ui_manager.ps, ui_manager.arrows)