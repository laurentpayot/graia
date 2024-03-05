from dataclasses import dataclass
import subprocess
import numpy as np

# Graia compilation
subprocess.run(["mkdir", "-p", "lib"])
subprocess.run(["futhark", "pyopencl", "--library", "-o", "lib/graia", "graia.fut"])
import lib.graia

print("\nInitializing Graia…")
g = lib.graia.graia()
print("Graia initialized.\n")

@dataclass
class Graia:
    layer_nb: int
    layer_size: int
    def fit (self, epoch: int) -> int: return self.layer_size + self.layer_nb * epoch + g.fit(2)

model = Graia(1, 2)
print(model.fit(10))
