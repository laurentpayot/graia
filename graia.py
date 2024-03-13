from dataclasses import dataclass
import subprocess
import numpy as np

print("\n🌄 Graia v0.0.1\n")

print("Compiling Graia…")
subprocess.run(["mkdir", "-p", "lib"])
subprocess.run(["touch", "lib/__init__.py"])
subprocess.run(["futhark", "pyopencl", "--library", "-o", "lib/graia", "graia.fut"])
from lib import graia

print("Initializing Graia…")
g = graia.graia()
print("Graia ready.\n")

@dataclass
class Graia:
    layer_nb: int
    layer_size: int
    def fit (self, epoch: int) -> int: return self.layer_size + self.layer_nb * epoch + g.fit(2)

# model = Graia(1, 2)
# print(model.fit(10))
