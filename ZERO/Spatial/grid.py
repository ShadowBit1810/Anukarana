import taichi as ti
import ZERO.Main.config as config

@ti.data_oriented
class SparseGrid:
    def __init__(self):
        # Configuration from config.py [cite: 6]
        self.dim = config.DIMENSION 
        self.res = config.GRID_RES
        self.block_size = config.BLOCK_SIZE 
        
        # Fields for Physics and ID tracking [cite: 4, 5]
        self.density = ti.field(dtype=config.PRECISION)
        self.voxel_id = ti.field(dtype=ti.i32) # Unique Voxel/Chunk ID
        
        # Setup the Hierarchy [cite: 4]
        self.indices = ti.ij if self.dim == 2 else ti.ijk
        self.offset = self.res // self.block_size
        
        # SNode Tree: Root -> Pointer (Sparse Chunks) -> Bitmask (Local Voxels)
        self.root = ti.root.pointer(self.indices, self.offset)
        self.block = self.root.bitmask(self.indices, self.block_size)
        
        # Place data to save VRAM (only allocated on demand) 
        self.block.place(self.density, self.voxel_id)
    
    @ti.func
    def get_v_id(self, I):
        """Generates a unique ID for each voxel across the global grid."""
        return I[0] + I[1] * self.res + (I[2] * self.res**2 if self.dim == 3 else 0)

    @ti.func
    def deposit_mass(self, p_pos, p_mass):
        """P2G Mass Deposition with boundary safety."""
        # Convert world position to grid index [cite: 4]
        grid_pos = p_pos * (1.0 / config.DX) 
        base = int(grid_pos - 0.5)
        fx = grid_pos - float(base)

        # Quadratic weights for smooth distribution [cite: 15]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        # Overlap distribution (3x3 for 2D, 3x3x3 for 3D)
        for offset in ti.static(ti.grouped(ti.ndrange(*( [3] * self.dim)))):
            idx = base + offset
            # Ghost Cell Safety: Ensure index is within grid bounds
            if all(idx >= 0) and all(idx < self.res):
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                
                # Atomic add to sparse field
                self.density[idx] += weight * p_mass
                self.voxel_id[idx] = self.get_v_id(idx)

    @ti.kernel
    def update_active_chunks(self):
        """Deactivates empty blocks to free VRAM[cite: 3, 8]."""
        for I in ti.grouped(self.block):
            # Check if this specific voxel has meaningful density
            if ti.is_active(self.block, I):
                if self.density[I] < 1e-9:
                    # Logic: Only deactivate if the entire chunk is empty
                    # For simplicity, we sniff the local voxel here
                    ti.deactivate(self.block, I)

    @ti.kernel
    def clear_grid(self):
        """Clears density but maintains the sparse structure for the current step."""
        for I in ti.grouped(self.density):
            self.density[I] = 0.0