import taichi as ti
from ZERO.Main import config

@ti.data_oriented
class ParticleState:
    def __init__(self, dim, n, precision = ti.f32, features = None) -> None:
        self.n = n
        self.dim = dim
        self.active_n = 0
        self.features = features if features is not None else []
        self.pos = ti.Vector.field(self.dim, dtype= precision)
        self.vel = ti.Vector.field(self.dim, dtype = precision)
        self.force = ti.Vector.field(self.dim, dtype= ti.f32)
        self.mat_id = ti.field(dtype = ti.i16)

        self.mass = None
        if "gravity" in self.features: self.mass = ti.field(dtype = precision)

        self.density = None
        self.pressure = None
        if "density" in self.features: 
            self.density = ti.field(dtype = precision)
            self.pressure = ti.field(dtype = precision)

        self._allocate_memory()
    
    def _allocate_memory(self):
        # On a 2GB VRAM GPU, we use a 1D dense block for particles 
        # This creates a Struct-of-Arrays (SoA) for better cache speed 
        block = ti.root.dense(ti.i, self.n)
        
        # Place core attributes
        block.place(self.pos, self.vel, self.mat_id, self.force)
        
        # Place optional attributes [cite: 16]
        if "gravity" in self.features: 
            block.place(self.mass)
        if "density" in self.features: 
            block.place(self.density, self.pressure)

    @ti.kernel
    def reset_particles(self):
        for i in range(self.n):
            self.vel[i] = ti.Vector.zero(float, self.dim)
            # Tell Taichi to evaluate this condition at compile-time!
            if ti.static("density" in self.features):
                self.density[i] = 0.0