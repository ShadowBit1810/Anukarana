import taichi as ti

# --- System State (The 3 Stages) ---
# 1: Input/Menu, 2: Frozen Sandbox, 3: Physics Collapse
STATE_MENU = 0
STATE_SANDBOX = 1
STATE_SIM = 2
CURRENT_STAGE = 0

# PRECISION - Strictly 32-bit for MX350 2GB VRAM 
PRECISION = ti.f32 
PREFERRED_ARCH = ti.gpu

# |----Simulation Constant----|
# These will be overwritten by the Launcher/Menu
DEFAULT_PARTICLES = 1000
MAX_PARTICLE = 15000
ACTIVE_PARTICLES = 1000
DIMENSION = 3         # or 2, Added: crucial for ParticleState init
G_CONSTANT = 1.0
SOFTENING = 1e-6
DT = 1/60.0
TIME_SCALE = 1.0   # multiplier: 1.0 = realtime, 0.0 = paused, 5.0 = 5x faster

# --- Sparse Grid Constants (The 2GB VRAM Saver) ---
# Using a Bitmask SNode helps fit a massive project in 2GB [cite: 4]
GRID_RES = 128
BLOCK_SIZE = 8
GRID_X = 16
GRID_Y = 16
GRID_Z = 16
GRID_SPACING = 0.02

# |----VRAM Budgeting----|
# MX350 has 2GB. We reserve 1.2GB for Taichi, leaving 0.8GB for Windows/UI.
MAX_VRAM_GB = 0.8

# |----Visuals----|
WINDOW_RES = (1280, 720)
PARTICLE_RADIUS = 0.005
MAIN_COLOR  = (0.2, 0.8, 0.5)
ARROW_COLOR = (1.0, 0.6, 0.1)   # warm orange — distinct from particle green
FPS = 60
BG_COLOR = (0.1, 0.1, 0.1)

# ---Initialization----
def initialise_taichi():
    """At starting of program,"""
    ti.init(
        arch=PREFERRED_ARCH,
        default_fp=PRECISION,
        device_memory_GB=MAX_VRAM_GB,
        fast_math=True,
        offline_cache=True
    )
    print(f"Taichi initialized on {PREFERRED_ARCH} with {MAX_VRAM_GB}GB VRAM.")