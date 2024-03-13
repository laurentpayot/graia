from typing import List
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


class Graia:
    def __init__(self, inputs: int, layers: int, layer_neurons: int, neuron_dendrites: int, outputs: int):
        self.config : dict = {
            "inputs": inputs,
            "layers": layers,
            "layer_neurons": layer_neurons,
            "neuron_dentrites": neuron_dendrites,
            "outputs": outputs
        }
        # TODO
        self.positive_weights: np.array[np.array[np.bool_]] = np.array([[0]], dtype=np.bool_)
        self.negative_weights: np.array[np.array[np.bool_]] = np.array([[0]], dtype=np.bool_)

    def fit (self, xs: np.array[np.array[np.uint8]], ys: np.array[np.uint8], epochs: int) -> int:
        return 42


# model = Graia(4, 2)
# print(model.fit(10))
