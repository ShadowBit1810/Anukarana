"""
Velocity arrow renderer using Taichi scene.lines().

Each arrow = 5 line segments = 10 vertices:
  [0,1]  shaft base → tip
  [2,3]  tip → head wing  +p1
  [4,5]  tip → head wing  -p1
  [6,7]  tip → head wing  +p2
  [8,9]  tip → head wing  -p2

p1 and p2 are two perpendicular axes to the velocity direction,
giving the arrowhead a cross/diamond shape visible from any camera angle.
"""

import taichi as ti
from ZERO.Main import config

VERTS_PER_ARROW = 10   # 5 segments × 2 endpoints


@ti.data_oriented
class ArrowRenderer:
    def __init__(self, ps):
        self.ps         = ps
        self.visible    = True

        # ── Appearance (tunable from sandbox UI) ────────────────────────────
        # arrow_scale : converts speed (m/s) → world-space arrow length
        #   e.g. scale=0.00005 → speed 1000 m/s draws an arrow 0.05 units long
        self.arrow_scale  = ti.field(dtype=ti.f32, shape=())
        self.head_ratio   = ti.field(dtype=ti.f32, shape=())  # head as % of length
        self.head_spread  = ti.field(dtype=ti.f32, shape=())  # wing width multiplier

        self.arrow_scale[None]  = 0.00005
        self.head_ratio[None]   = 0.25
        self.head_spread[None]  = 0.4

        # ── Vertex buffer ────────────────────────────────────────────────────
        # Sized for ALL allocated particles so the field never needs realloc.
        # Inactive particles get degenerate (zero-length) lines → invisible.
        self.verts = ti.Vector.field(3, dtype=ti.f32, shape=ps.n * VERTS_PER_ARROW)

    # ────────────────────────────────────────────────────────────────────────
    # Kernel — fills vertex buffer every frame
    # ────────────────────────────────────────────────────────────────────────
    @ti.kernel
    def update(self, active_n: ti.i32):
        scale       = self.arrow_scale[None]
        head_ratio  = self.head_ratio[None]
        head_spread = self.head_spread[None]

        for i in range(self.ps.n):
            base = i * VERTS_PER_ARROW

            if i < active_n:
                pos   = self.ps.pos[i]
                vel   = self.ps.vel[i]
                speed = vel.norm()

                if speed > 1e-3:
                    direction  = vel / speed
                    arrow_len  = speed * scale
                    tip        = pos + direction * arrow_len
                    head_start = pos + direction * arrow_len * (1.0 - head_ratio)
                    hw         = arrow_len * head_spread

                    # ── Two perpendicular axes for the 3D arrowhead ─────────
                    # Primary: cross with world-up; fallback to world-right
                    # when velocity is nearly vertical (avoids degenerate cross)
                    world_up = ti.Vector([0.0, 1.0, 0.0])
                    if ti.abs(direction.dot(world_up)) > 0.99:
                        world_up = ti.Vector([1.0, 0.0, 0.0])
                    p1 = direction.cross(world_up).normalized()
                    p2 = direction.cross(p1).normalized()

                    # Shaft
                    self.verts[base + 0] = pos
                    self.verts[base + 1] = tip
                    # Head wings (cross pattern)
                    self.verts[base + 2] = tip
                    self.verts[base + 3] = head_start + p1 * hw
                    self.verts[base + 4] = tip
                    self.verts[base + 5] = head_start - p1 * hw
                    self.verts[base + 6] = tip
                    self.verts[base + 7] = head_start + p2 * hw
                    self.verts[base + 8] = tip
                    self.verts[base + 9] = head_start - p2 * hw

                else:
                    # Near-zero velocity → collapse all segments to pos (invisible)
                    for k in ti.static(range(VERTS_PER_ARROW)):
                        self.verts[base + k] = pos

            else:
                # Inactive particle → park far off-screen
                for k in ti.static(range(VERTS_PER_ARROW)):
                    self.verts[base + k] = ti.Vector([-1000.0, -1000.0, -1000.0])