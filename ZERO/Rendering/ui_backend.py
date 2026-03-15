import taichi as ti
import numpy as np
import ZERO.Main.config as config
from ZERO.State.particle import ParticleState
from ZERO.Rendering.arrow_renderer import ArrowRenderer

# ─── Place-mode constants ───────────────────────────────────────────────────
MODE_GRID    = 0
MODE_SINGLE  = 1
MODE_CLUSTER = 2


@ti.data_oriented
class WindowManager:
    def __init__(self):
        self.window = ti.ui.Window("ZERO Physics Simulator", config.WINDOW_RES, vsync=True)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color(config.BG_COLOR)
        # In WindowManager.__init__
        self.scene  = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0.5, 0.5, 2.0)   # where the eye is
        self.camera.lookat(0.5, 0.5, 0.5)     # what it looks at

    def render_screen(self, ps, arrows: ArrowRenderer = None):
        self.camera.track_user_inputs(
            self.window, movement_speed=0.03, hold_key=ti.ui.RMB
        )
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.4, 0.4, 0.4))
        self.scene.point_light(pos=(1, 2, 2), color=(1, 1, 1))

        if ps is not None:
            self.scene.particles(
                ps.pos,
                radius=config.PARTICLE_RADIUS,
                color=config.MAIN_COLOR,
            )

        # Draw velocity arrows when renderer exists and is toggled on
        if arrows is not None and arrows.visible:
            self.scene.lines(
                arrows.verts,
                width=1.5,
                color=config.ARROW_COLOR,
            )

        self.canvas.scene(self.scene)
        self.window.show()


@ti.data_oriented
class UIManager:
    def __init__(self):
        self.Win = WindowManager()
        self.ps: ParticleState = None   # created after menu
        self.arrows: ArrowRenderer = None  # created alongside ps

        # ── Menu temporaries ────────────────────────────────────────────────
        self.temp_active_particles = config.DEFAULT_PARTICLES
        self.use_gravity  = False
        self.use_density  = False

        # ── Grid-layout panel ───────────────────────────────────────────────
        self.grid_x       = config.GRID_X
        self.grid_y       = config.GRID_Y
        self.grid_z       = config.GRID_Z
        self.grid_spacing = config.GRID_SPACING

        # ── Particle-editor state ────────────────────────────────────────────
        self.place_mode = MODE_GRID      # which sub-panel is active

        # Single-particle editor
        self.sel_idx   = 0              # chosen particle index (spinbox value)
        self.sel_x     = 0.5
        self.sel_y     = 0.5
        self.sel_z     = 0.5
        self.sel_mass  = 1.0            # mass of selected particle (gravity only)
        self.sel_vx    = 0.0            # velocity components of selected particle
        self.sel_vy    = 0.0
        self.sel_vz    = 0.0

        # Cluster editor
        self.cl_start  = 0
        self.cl_end    = 0
        self.cl_locked = False          # True after "Lock Cluster" is pressed
        self.cl_dx     = 0.0           # live offset from locked snapshot
        self.cl_dy     = 0.0
        self.cl_dz     = 0.0
        self.cl_mass   = 1.0            # uniform mass to apply to cluster (gravity only)
        self.cl_vx     = 0.0            # velocity to apply to cluster
        self.cl_vy     = 0.0
        self.cl_vz     = 0.0

        # Snapshot field — allocated once ps is ready (see _init_snapshot_field)
        self._snap_field: ti.template() = None   # ti.Vector.field(3, f32, MAX)
        self._snap_np: np.ndarray = None          # python-side mirror

    # ────────────────────────────────────────────────────────────────────────
    # Snapshot helpers
    # ────────────────────────────────────────────────────────────────────────
    def _init_snapshot_field(self):
        """Call once after ps is initialised."""
        self._snap_field = ti.Vector.field(3, dtype=ti.f32, shape=self.ps.n)

    def _lock_cluster_snapshot(self):
        """Capture current particle positions for the selected cluster range."""
        self._snap_np = self.ps.pos.to_numpy().copy()   # shape (n, 3)
        self._snap_field.from_numpy(self._snap_np)
        self.cl_dx = self.cl_dy = self.cl_dz = 0.0
        self.cl_locked = True

    def _unlock_cluster(self):
        """Commit offsets, bake positions, release lock."""
        self.cl_locked = False
        self.cl_dx = self.cl_dy = self.cl_dz = 0.0

    # ────────────────────────────────────────────────────────────────────────
    # Main draw dispatcher
    # ────────────────────────────────────────────────────────────────────────
    def draw(self):
        gui = self.Win.window.get_gui()

        if config.CURRENT_STAGE == config.STATE_MENU:
            self._draw_menu(gui)
        elif config.CURRENT_STAGE == config.STATE_SANDBOX:
            # Live single-particle drag — push every frame, same as cluster
            if self.place_mode == MODE_SINGLE and self.ps is not None:
                self._place_single_particle(
                    self.sel_idx,
                    self.sel_x, self.sel_y, self.sel_z,
                )
                # Live mass push — only if gravity feature is active
                if "gravity" in self.ps.features:
                    self._set_single_mass(self.sel_idx, self.sel_mass)
                # Live velocity push — always (vel is a core field)
                self._set_single_vel(
                    self.sel_idx,
                    self.sel_vx, self.sel_vy, self.sel_vz,
                )
            # Live cluster drag — apply snapshot + offset every frame
            if self.cl_locked:
                self._apply_cluster_offset(
                    self.cl_start, self.cl_end,
                    self.cl_dx, self.cl_dy, self.cl_dz,
                )
            # Rebuild arrow vertices from current pos/vel
            if self.arrows is not None:
                self.arrows.update(self.ps.active_n)
            self._draw_sandbox(gui)
        elif config.CURRENT_STAGE == config.STATE_SIM:
            # Keep arrows updated during simulation too
            if self.arrows is not None:
                self.arrows.update(self.ps.active_n)
            self._draw_sim(gui)

    # ────────────────────────────────────────────────────────────────────────
    # Stage 0 – Menu
    # ────────────────────────────────────────────────────────────────────────
    def _draw_menu(self, gui):
        with gui.sub_window("Simulation Setup", 0.05, 0.05, 0.4, 0.55):
            gui.text(f"Max Particles allowed: {config.MAX_PARTICLE}")
            self.temp_active_particles = gui.slider_int(
                "Active Particles", self.temp_active_particles,
                1, config.MAX_PARTICLE,
            )
            self.use_gravity = gui.checkbox("Enable Gravity",       self.use_gravity)
            self.use_density = gui.checkbox("Enable Fluid Density", self.use_density)

            if gui.button("Generate Sandbox"):
                config.ACTIVE_PARTICLES = self.temp_active_particles
                features = []
                if self.use_gravity: features.append("gravity")
                if self.use_density: features.append("density")

                self.ps = ParticleState(
                    dim=config.DIMENSION, n=config.ACTIVE_PARTICLES,
                    precision=config.PRECISION, features=features,
                )
                self.arrows = ArrowRenderer(self.ps)
                self._init_snapshot_field()

                # Clamp editor indices to valid range
                self.sel_idx = 0
                self.cl_start = 0
                self.cl_end   = min(99, self.ps.n - 1)

                self._arrange_and_update_active()
                config.CURRENT_STAGE = config.STATE_SANDBOX

    # ────────────────────────────────────────────────────────────────────────
    # Stage 1 – Sandbox
    # ────────────────────────────────────────────────────────────────────────
    def _draw_sandbox(self, gui):
        n = self.ps.n

        # ── LEFT: Grid layout panel ─────────────────────────────────────────
        with gui.sub_window("Grid Layout", 0.02, 0.05, 0.28, 0.55):
            gui.text(f"Allocated Particles: {n}")
            max_axis = int(n ** (1 / 3)) + 5

            new_x = gui.slider_int("Grid X", self.grid_x, 1, max_axis)
            new_y = gui.slider_int("Grid Y", self.grid_y, 1, max_axis)
            new_z = gui.slider_int("Grid Z", self.grid_z, 1, max_axis)
            new_sp = gui.slider_float("Spacing", self.grid_spacing, 0.005, 0.1)

            if (new_x != self.grid_x or new_y != self.grid_y
                    or new_z != self.grid_z or new_sp != self.grid_spacing):
                self.grid_x, self.grid_y = new_x, new_y
                self.grid_z, self.grid_spacing = new_z, new_sp
                if self.cl_locked:
                    self.cl_locked = False          # stale snapshot, release
                self._arrange_and_update_active()

            used = self.grid_x * self.grid_y * self.grid_z
            if used > n:
                gui.text(f"! Grid ({used}) exceeds allocated ({n})")
                gui.text("  Extra slots hidden off-screen.")
            else:
                gui.text(f"Using {used} / {n} particles")

        # ── RIGHT: Particle editor ──────────────────────────────────────────
        with gui.sub_window("Particle Editor", 0.32, 0.05, 0.66, 0.85):

            # ---- mode selector (button row) --------------------------------
            gui.text("─── Mode ───────────────────────")
            if gui.button("[Grid Reset]"):
                self.place_mode = MODE_GRID
                if self.cl_locked:
                    self.cl_locked = False
                self._arrange_and_update_active()

            if gui.button("[Single Particle]"):
                self.place_mode = MODE_SINGLE
                self.cl_locked  = False

            if gui.button("[Cluster Select]"):
                self.place_mode = MODE_CLUSTER
                self.cl_locked  = False

            gui.text(
                f"Active: {'GRID' if self.place_mode == MODE_GRID else 'SINGLE' if self.place_mode == MODE_SINGLE else 'CLUSTER'}"
            )
            gui.text("────────────────────────────────")

            # ---- single-particle editor ------------------------------------
            if self.place_mode == MODE_SINGLE:
                self._panel_single(gui, n)

            # ---- cluster editor --------------------------------------------
            elif self.place_mode == MODE_CLUSTER:
                self._panel_cluster(gui, n)

            # ---- grid mode (info only) -------------------------------------
            else:
                gui.text("Adjust sliders in 'Grid Layout'")
                gui.text("to re-arrange all particles.")

        # ── Arrow controls ──────────────────────────────────────────────────
        with gui.sub_window("Velocity Arrows", 0.02, 0.62, 0.28, 0.24):
            self.arrows.visible = gui.checkbox("Show Arrows", self.arrows.visible)

            if self.arrows.visible:
                # Scale slider — log-feel by using a small float range
                cur_scale = self.arrows.arrow_scale[None]
                new_scale = gui.slider_float("Scale", cur_scale, 0.000001, 0.001)
                if new_scale != cur_scale:
                    self.arrows.arrow_scale[None] = new_scale

                cur_spread = self.arrows.head_spread[None]
                new_spread = gui.slider_float("Head Width", cur_spread, 0.05, 1.0)
                if new_spread != cur_spread:
                    self.arrows.head_spread[None] = new_spread

        # ── BOTTOM: Start simulation ────────────────────────────────────────
        with gui.sub_window("Controls", 0.02, 0.88, 0.96, 0.10):
            if gui.button(">>> START SIMULATION <<<"):
                if self.cl_locked:
                    self.cl_locked = False
                config.CURRENT_STAGE = config.STATE_SIM

    # ── Single-particle panel ────────────────────────────────────────────────
    def _panel_single(self, gui, n: int):
        gui.text("Select a particle — position updates live.")
        gui.text("")

        # Spinbox emulation: slider_int over [0, n-1]
        new_idx = gui.slider_int("Particle #", self.sel_idx, 0, n - 1)
        if new_idx != self.sel_idx:
            self.sel_idx = new_idx
            # ONE batched numpy read — only triggers on index change, not every frame
            pos_np = self.ps.pos.to_numpy()
            vel_np = self.ps.vel.to_numpy()
            self.sel_x  = float(pos_np[self.sel_idx, 0])
            self.sel_y  = float(pos_np[self.sel_idx, 1])
            self.sel_z  = float(pos_np[self.sel_idx, 2]) if config.DIMENSION == 3 else 0.5
            self.sel_vx = float(vel_np[self.sel_idx, 0])
            self.sel_vy = float(vel_np[self.sel_idx, 1])
            self.sel_vz = float(vel_np[self.sel_idx, 2]) if config.DIMENSION == 3 else 0.0
            if "gravity" in self.ps.features:
                mass_np = self.ps.mass.to_numpy()
                self.sel_mass = float(mass_np[self.sel_idx])

        # Live position sliders — particle follows instantly (pushed in draw())
        self.sel_x = gui.slider_float("X", self.sel_x, 0.0, 1.0)
        self.sel_y = gui.slider_float("Y", self.sel_y, 0.0, 1.0)
        self.sel_z = gui.slider_float("Z", self.sel_z, 0.0, 1.0)

        # Mass slider — only visible when gravity feature is enabled
        if "gravity" in self.ps.features:
            gui.text("")
            gui.text("── Mass ──────────────────────────")
            self.sel_mass = gui.slider_float("Mass", self.sel_mass, 0.1, 100.0)

        # Velocity sliders — always shown, live push via draw()
        gui.text("")
        gui.text("── Velocity (m/s) ────────────────")
        self.sel_vx = gui.slider_float("Vx", self.sel_vx, -10000.0, 10000.0)
        self.sel_vy = gui.slider_float("Vy", self.sel_vy, -10000.0, 10000.0)
        self.sel_vz = gui.slider_float("Vz", self.sel_vz, -10000.0, 10000.0)
        if gui.button("Zero Velocity"):
            self.sel_vx = self.sel_vy = self.sel_vz = 0.0

        gui.text("")

        if gui.button("Randomize This Particle"):
            self._randomize_range(self.sel_idx, self.sel_idx)
            pos_np = self.ps.pos.to_numpy()
            self.sel_x = float(pos_np[self.sel_idx, 0])
            self.sel_y = float(pos_np[self.sel_idx, 1])
            self.sel_z = float(pos_np[self.sel_idx, 2])

    # ── Cluster panel ────────────────────────────────────────────────────────
    def _panel_cluster(self, gui, n: int):
        gui.text("Select a range, lock it, then drag.")
        gui.text("")

        # Range selection
        new_start = gui.slider_int("Start #", self.cl_start, 0, n - 1)
        new_end   = gui.slider_int("End   #", self.cl_end,   0, n - 1)

        # Keep start <= end
        if new_start > new_end:
            new_end = new_start
        if new_end < new_start:
            new_start = new_end

        range_changed = (new_start != self.cl_start or new_end != self.cl_end)
        self.cl_start = new_start
        self.cl_end   = new_end

        count = self.cl_end - self.cl_start + 1
        gui.text(f"  {count} particles selected  [{self.cl_start} … {self.cl_end}]")
        gui.text("")

        # If range changed, unlock so user must re-lock
        if range_changed and self.cl_locked:
            self.cl_locked = False
            gui.text("! Range changed — re-lock to move.")

        # Lock / unlock toggle
        if not self.cl_locked:
            if gui.button("Lock Cluster  (enables drag)"):
                self._lock_cluster_snapshot()
        else:
            gui.text("Cluster LOCKED  — drag below:")
            gui.text("")
            self.cl_dx = gui.slider_float("Offset X", self.cl_dx, -0.5, 0.5)
            self.cl_dy = gui.slider_float("Offset Y", self.cl_dy, -0.5, 0.5)
            self.cl_dz = gui.slider_float("Offset Z", self.cl_dz, -0.5, 0.5)

            # Mass for the cluster — only when gravity is enabled
            if "gravity" in self.ps.features:
                gui.text("")
                gui.text("── Mass ──────────────────────────")
                self.cl_mass = gui.slider_float("Set Mass", self.cl_mass, 0.1, 100.0)
                if gui.button("Apply Mass to Cluster"):
                    self._set_cluster_mass(self.cl_start, self.cl_end, self.cl_mass)

            # Velocity for the cluster — always shown
            gui.text("")
            gui.text("── Velocity (m/s) ────────────────")
            self.cl_vx = gui.slider_float("Set Vx", self.cl_vx, -10000.0, 10000.0)
            self.cl_vy = gui.slider_float("Set Vy", self.cl_vy, -10000.0, 10000.0)
            self.cl_vz = gui.slider_float("Set Vz", self.cl_vz, -10000.0, 10000.0)
            if gui.button("Apply Velocity to Cluster"):
                self._set_cluster_vel(
                    self.cl_start, self.cl_end,
                    self.cl_vx, self.cl_vy, self.cl_vz,
                )
            if gui.button("Zero Cluster Velocity"):
                self.cl_vx = self.cl_vy = self.cl_vz = 0.0
                self._set_cluster_vel(self.cl_start, self.cl_end, 0.0, 0.0, 0.0)

            gui.text("")

            if gui.button("Apply & Unlock"):
                # Bake current offset into positions, then release
                self._apply_cluster_offset(
                    self.cl_start, self.cl_end,
                    self.cl_dx, self.cl_dy, self.cl_dz,
                )
                self._unlock_cluster()

            if gui.button("Randomize Cluster"):
                self._randomize_range(self.cl_start, self.cl_end)
                self.cl_locked = False   # snapshot is now stale

            if gui.button("Cancel / Restore Positions"):
                # Roll back by re-applying zero offset from snapshot
                self._apply_cluster_offset(
                    self.cl_start, self.cl_end, 0.0, 0.0, 0.0,
                )
                self._unlock_cluster()

    # ────────────────────────────────────────────────────────────────────────
    # Stage 2 – Simulation (collapsed)
    # ────────────────────────────────────────────────────────────────────────
    def _draw_sim(self, gui):
        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.28):
            if gui.button("Reset to Sandbox"):
                self.ps.reset_particles()
                self._arrange_and_update_active()
                self.cl_locked = False
                config.TIME_SCALE = 1.0
                config.CURRENT_STAGE = config.STATE_SANDBOX

            gui.text("")
            gui.text("── Time Scale ───────────────────")
            config.TIME_SCALE = gui.slider_float(
                "Speed", config.TIME_SCALE, 0.0, 5.0
            )
            gui.text(f"  DT = {config.DT * config.TIME_SCALE:.5f} s")

            # Pause toggle — TIME_SCALE=0 means gravity integrates zero displacement
            is_paused = config.TIME_SCALE == 0.0
            if gui.button("Resume" if is_paused else "Pause"):
                config.TIME_SCALE = 1.0 if is_paused else 0.0

    # ────────────────────────────────────────────────────────────────────────
    # Taichi kernels
    # ────────────────────────────────────────────────────────────────────────

    @ti.kernel
    def arrange_particles_grid(self):
        """Fill ps.pos with a 3-D grid; hide surplus particles off-screen."""
        total_grid = self.grid_x * self.grid_y * self.grid_z
        offset_x = 0.5 - (self.grid_x * self.grid_spacing) / 2.0
        offset_y = 0.5 - (self.grid_y * self.grid_spacing) / 2.0
        offset_z = 0.5 - (self.grid_z * self.grid_spacing) / 2.0

        for i in range(self.ps.n):
            if i < total_grid:
                ix = i % self.grid_x
                iy = (i // self.grid_x) % self.grid_y
                iz = i // (self.grid_x * self.grid_y)
                self.ps.pos[i] = ti.Vector(
                    [offset_x + ix * self.grid_spacing,
                     offset_y + iy * self.grid_spacing,
                     offset_z + iz * self.grid_spacing],
                    dt=config.PRECISION,
                )
                self.ps.vel[i] = ti.Vector.zero(config.PRECISION, config.DIMENSION)
            else:
                # Park unused particles far off-screen
                self.ps.pos[i] = ti.Vector([-1000.0, -1000.0, -1000.0],
                                            dt=config.PRECISION)
                self.ps.vel[i] = ti.Vector.zero(config.PRECISION, config.DIMENSION)
    
    def _arrange_and_update_active(self):
        self.arrange_particles_grid()
        self.ps.active_n = min(
            self.grid_x * self.grid_y * self.grid_z,
            self.ps.n
        )

    @ti.kernel
    def _place_single_particle(self, idx: ti.i32,
                                x: ti.f32, y: ti.f32, z: ti.f32):
        """Move one particle to (x, y, z) and zero its velocity."""
        self.ps.pos[idx] = ti.Vector([x, y, z], dt=config.PRECISION)
        self.ps.vel[idx] = ti.Vector.zero(config.PRECISION, config.DIMENSION)

    @ti.kernel
    def _randomize_range(self, start: ti.i32, end: ti.i32):
        """Place each particle in [start, end] at a random position in (0,1)^3."""
        for i in range(start, end + 1):
            self.ps.pos[i] = ti.Vector(
                [ti.random(config.PRECISION),
                 ti.random(config.PRECISION),
                 ti.random(config.PRECISION)],
                dt=config.PRECISION,
            )
            self.ps.vel[i] = ti.Vector.zero(config.PRECISION, config.DIMENSION)

    @ti.kernel
    def _apply_cluster_offset(self, start: ti.i32, end: ti.i32,
                               dx: ti.f32, dy: ti.f32, dz: ti.f32):
        """
        Move cluster particles to (snapshot_pos + offset).
        Called every frame while cl_locked==True for live drag feedback.
        """
        for i in range(start, end + 1):
            base = self._snap_field[i]
            self.ps.pos[i] = ti.Vector(
                [base[0] + dx, base[1] + dy, base[2] + dz],
                dt=config.PRECISION,
            )

    @ti.kernel
    def _set_single_mass(self, idx: ti.i32, mass: ti.f32):
        """Set mass of one particle. Only call when gravity feature is active."""
        self.ps.mass[idx] = mass

    @ti.kernel
    def _set_cluster_mass(self, start: ti.i32, end: ti.i32, mass: ti.f32):
        """Set uniform mass for every particle in [start, end]."""
        for i in range(start, end + 1):
            self.ps.mass[i] = mass

    @ti.kernel
    def _set_single_vel(self, idx: ti.i32, vx: ti.f32, vy: ti.f32, vz: ti.f32):
        """Set velocity of one particle. Live — called every frame in draw()."""
        self.ps.vel[idx] = ti.Vector([vx, vy, vz], dt=config.PRECISION)

    @ti.kernel
    def _set_cluster_vel(self, start: ti.i32, end: ti.i32,
                          vx: ti.f32, vy: ti.f32, vz: ti.f32):
        """Set uniform velocity for every particle in [start, end]."""
        for i in range(start, end + 1):
            self.ps.vel[i] = ti.Vector([vx, vy, vz], dt=config.PRECISION)