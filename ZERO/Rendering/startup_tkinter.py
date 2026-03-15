"""Making start up window to initialise the taichi kernal of how much ram with max particles. 
And how much max particles needed to start the program
And on which backend needed to be runned"""

import tkinter as tk
from tkinter import ttk, messagebox
from ZERO.Main import config

class Screen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ZERO Engine Launcher")
        self.root.geometry("400x350")
        self.root.configure(bg='#1e1e1e') # Dark background
        
        self._setup_styles()
        self._entry_data()
        self.launched = False   # flag to track if user clicked Launch
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

        if not self.launched:
            import sys
            sys.exit(0)

    def _on_close(self):
        """User closed the window without launching."""
        self.root.destroy()
    
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", foreground="white", background="#1e1e1e", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))

    def _entry_data(self):
        # Header
        tk.Label(self.root, text="INITIALIZATION SETTINGS", bg='#1e1e1e', fg='#00ffcc', 
                 font=("Segoe UI", 14, "bold")).pack(pady=10)

        # Backend Selection (Radio Buttons)
        self.backend_var = tk.IntVar(value=1)
        frame_backend = tk.Frame(self.root, bg='#1e1e1e')
        frame_backend.pack(pady=5)
        tk.Radiobutton(frame_backend, text="GPU (Vulkan/Metal)", variable=self.backend_var, value=1,
                       bg='#1e1e1e', fg='white', selectcolor='#333333').pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(frame_backend, text="CPU", variable=self.backend_var, value=2,
                       bg='#1e1e1e', fg='white', selectcolor='#333333').pack(side=tk.LEFT, padx=10)

        # Particle Inputs with Entry Boxes
        self.max_p = self._create_input_row("Max Particles (Capacity):", config.MAX_PARTICLE)
        
        # GPU Memory Cap (Crucial for your 2GB VRAM)
        self.vram_cap = self._create_input_row("VRAM Limit (GB):", 1.0)

        # Launch Button
        btn_launch = tk.Button(self.root, text="INITIALIZE ENGINE", command=self.launch,
                               bg='#00ffcc', fg='#1e1e1e', font=("Segoe UI", 11, "bold"),
                               activebackground='#00ccaa', width=20)
        btn_launch.pack(pady=20)

    def _create_input_row(self, label_text, default_val):
        frame = tk.Frame(self.root, bg='#1e1e1e')
        frame.pack(fill='x', padx=40, pady=5)
        tk.Label(frame, text=label_text, bg='#1e1e1e', fg='white').pack(side=tk.LEFT)
        entry = tk.Entry(frame, width=10, bg='#333333', fg='white', insertbackground='white')
        entry.insert(0, str(default_val))
        entry.pack(side=tk.RIGHT)
        return entry

    
    def launch(self):
       try:
           max_val = int(self.max_p.get())
           if max_val > 2000000:
               if not messagebox.askyesno("Warning", "High particle count may crash your GPU. Proceed?"):
                   return

           config.MAX_PARTICLE = max_val
           config.BACKEND = "GPU" if self.backend_var.get() == 1 else "CPU"

           self.launched = True   # ← mark as properly launched
           self.root.destroy()
       except ValueError:
           messagebox.showerror("Error", "Please enter valid numbers.")