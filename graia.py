# from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import subprocess

print("\n🌄 Graia v0.0.1\n")

print("Graia compiling…")
subprocess.run(["mkdir", "-p", "lib"])
subprocess.run(["touch", "lib/__init__.py"])
subprocess.run(["futhark", "pyopencl", "--library", "-o", "lib/graia", "graia.fut"])

from lib import graia

print("Graia initializing…")
g = graia.graia()
print("Graia ready.\n")


class Graia:
    def __init__(self,
                inputs: int,
                layers: int,
                layer_neurons: int,
                outputs: int,
                # TODO
                neuron_dendrites=0,
                ) -> None:

        self.parameters: int = (
            (inputs * layer_neurons) +
            (layer_neurons * layer_neurons * (layers - 1)) +
            (layer_neurons * outputs)
        )

        self.config : dict = {
            "inputs": inputs,
            "layers": layers,
            "layer_neurons": layer_neurons,
            "outputs": outputs,
            # "neuron_dentrites": neuron_dendrites,
        }

        # TODO
        self.input_weights: NDArray[np.int8] = np.array([[0],[0]], dtype=np.int8)
        self.hidden_weights: NDArray[np.int8] = np.array([[0],[0]], dtype=np.int8)
        self.output_weights: NDArray[np.int8] = np.array([[0],[0]], dtype=np.int8)
        print(f"Graia model with {self.parameters} parameters ready.")


    def fit (self, xs: NDArray[np.uint8], ys: NDArray[np.uint8], epochs: int):
        return g.fit(xs, ys, np.int32(epochs))
