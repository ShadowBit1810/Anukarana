# Anukarna — GPU Particle Physics Simulator
*अनुकरण — Sanskrit for "simulation"*

A real-time 3D particle physics simulator built from scratch in Python and Taichi.
Designed around constrained hardware (2GB VRAM), it uses explicit memory budgeting,
Struct-of-Arrays GPU layout, and an interactive sandbox to configure exact initial
conditions before physics runs.

Currently implements N-body gravitational simulation with O(N²/2) pair optimization
and symplectic Euler integration. SPH fluid and field physics are in active development.

---

## Features

- **Interactive Sandbox** — place, arrange and configure particles before simulation starts. Freeze time, set up your universe, then collapse it into physics
- **N-body Gravity** — Newton's law of gravitation with 3rd-law pair optimization (O(N²/2)), symplectic Euler integration
- **Single Particle Editor** — select any particle by index, set position, velocity and mass live with sliders
- **Cluster Editor** — select a range of particles, lock a snapshot, drag the whole group with preserved relative positions, apply uniform mass and velocity
- **3D Velocity Arrows** — real-time arrow rendering using `scene.lines()`, showing direction and magnitude of every particle's velocity. Two perpendicular arrowhead planes visible from any camera angle
- **Time Scale Control** — slow motion, pause, fast forward. Slider from 0x to 5x. Pause is free (integrates zero displacement, no special logic)
- **VRAM Budgeting** — configurable memory cap at launch. Designed and tested on MX350 (2GB VRAM)
- **Feature Gating** — mass, density and pressure fields only allocated when their physics feature is enabled. Unused particles hidden off-screen, excluded from all force calculations via `active_n`

---

## Architecture

```
Anukarna/
└── ZERO/
    ├── Main/
    │   ├── Sim.py              ← entry point and main loop
    │   └── config.py           ← all constants and shared state (module-level shared bus)
    │
    ├── State/
    │   └── particle.py         ← ParticleState, SoA field allocation, optional features
    │
    ├── Rendering/
    │   ├── ui_backend.py       ← UIManager (sandbox, editor), WindowManager (3D renderer)
    │   ├── arrow_renderer.py   ← velocity arrow GPU line builder
    │   └── startup_tkinter.py  ← launcher window (backend, VRAM, particle count)
    │
    └── Operation/
        └── gravity_ops.py      ← N-body gravity kernel
```

**Three-stage pipeline:**

```
STATE_MENU     →    STATE_SANDBOX    →    STATE_SIM
  (configure)        (arrange)             (physics)
```

Physics never runs during setup. You configure everything in a frozen sandbox,
then collapse into simulation. Returning to sandbox resets positions and
suspends the solver.

---

## Physics Notes

**Gravity** uses symplectic Euler integration — velocity is updated before position.
This conserves energy better than standard Euler for orbital mechanics, keeping
orbits stable over longer timescales.

Force computation uses Newton's 3rd law to halve pair evaluations:
```
for i in range(N):
    for j in range(i+1, N):   # each pair computed once
        force_ij = G * m_i * m_j / r²
        f[i] += force_ij      # pull i toward j
        f[j] -= force_ij      # pull j toward i  (free, same computation)
```

**Stable two-body orbit** — with `G=1`, separation `r=0.4`, mass `m=20`:
```
v = sqrt(m / 2r) = sqrt(20 / 0.8) = 5.0 units/s
```
Place particles at `(0.3, 0.5, 0.5)` and `(0.7, 0.5, 0.5)` with velocities
`Vy = +5.0` and `Vy = -5.0` respectively.

**Memory layout** uses Struct-of-Arrays (SoA) — all positions contiguous,
all velocities contiguous. GPU cache reads one field at a time, not
interleaved per-particle structs. Faster for kernel access patterns.

---

## Requirements

- Python 3.9+
- GPU with 2GB+ VRAM (CPU fallback selectable at launch)

```bash
pip install taichi
python ZERO/Main/Sim.py
```

---

## Controls

| Input | Action |
|---|---|
| Right mouse + drag | Orbit camera |
| Time Scale slider | 0x pause → 5x fast forward |
| Pause / Resume button | Toggle freeze |
| Single mode | Select particle by index, set position / velocity / mass live |
| Cluster mode | Lock range snapshot, drag group, apply uniform properties |
| Randomize | Scatter selected particles using `ti.random` |
| Reset to Sandbox | Return to frozen setup, reset positions |

---

## Launcher

On startup a tkinter window lets you configure:

- **Backend** — GPU (Vulkan/Metal) or CPU
- **Max Particles** — total allocated capacity (VRAM is reserved at this number)
- **VRAM Limit** — cap in GB, protects against OOM on low-VRAM cards

Closing the launcher without clicking Initialize shuts down the process cleanly
before Taichi ever touches the GPU.

---

## Hardware Target

Built and tested on:
- **GPU:** NVIDIA MX350 (2GB VRAM)
- **RAM:** 8GB
- **OS:** Windows 10
- **Python:** 3.13
- **Taichi:** 1.7.x

VRAM budget is configurable. The default reserves 0.8GB leaving headroom for
the OS and display driver.

---

## Roadmap

- [ ] SPH fluid simulation (density, pressure, viscosity)
- [ ] Field physics (gravitational potential grid, velocity fields)
- [ ] Collision detection
- [ ] Particle material types (different visual + physical properties)
- [ ] Export initial conditions to file, reload saved setups
- [ ] Packaged executable (PyInstaller)

---

## License

MIT — see [LICENSE](LICENSE)
