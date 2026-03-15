"""gravity operation required"""
from ZERO.Main import config
import taichi as ti

@ti.data_oriented
class gravity:
    def __init__(self, ps : ti.template):
        self.ps = ps
        # Do NOT call compute_gravity() here — mass is zeroed at construction

    @ti.kernel
    def compute_gravity(self, dt: ti.f32):
        # Reset forces to zero at the start of the frame
        for i in range(self.ps.active_n):
            self.ps.force[i] = ti.Vector([0.0, 0.0, 0.0])

        # O(N^2) Optimization: Compare each pair only once
        for i in range(self.ps.active_n):
            for j in range(i + 1, self.ps.active_n):   # ← was self.ps.n (bug fix)
                pos_i = self.ps.pos[i]
                pos_j = self.ps.pos[j]

                diff = pos_j - pos_i
                r_sq = diff.norm_sqr() + config.SOFTENING
                r = ti.sqrt(r_sq)

                # Force Magnitude: G * (m1 * m2) / r^2
                force_mag = config.G_CONSTANT * self.ps.mass[i] * self.ps.mass[j] / r_sq

                # Directional Force Vector: magnitude * (difference / distance)
                force_vec = force_mag * (diff / r)

                # Apply forces (Newton's 3rd Law)
                self.ps.force[i] += force_vec  # Pull i towards j
                self.ps.force[j] -= force_vec  # Pull j towards i (opposite direction)

        # Integration Step: F = ma -> a = F/m, v += a*dt, pos += v*dt
        for i in range(self.ps.active_n):
            acc = self.ps.force[i] / self.ps.mass[i]
            self.ps.vel[i] += acc * dt
            self.ps.pos[i] += self.ps.vel[i] * dt