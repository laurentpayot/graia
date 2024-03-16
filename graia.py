from typing import TypeAlias
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


# A weight of n is actually the inverse of 2 at the power of n (right shift by abs(n) - 1)
# Weights are negative for inhibition, positive for excitation, zero for no connection
Weight: TypeAlias = np.int8

InputVal: TypeAlias = np.uint8
OutputVal: TypeAlias = np.uint8

class Graia:
    def __init__(self,
                inputs: int,
                layer_neurons: int,
                layers: int,
                outputs: int,
                # TODO
                # neuron_dendrites=0,
                ) -> None:

        self.parameters: int = (
            (inputs * layer_neurons) +
            (layer_neurons * layer_neurons * (layers - 1)) +
            (layer_neurons * outputs)
        )

        self.config : dict = {
            "inputs": inputs,
            "layer_neurons": layer_neurons,
            "layers": layers,
            "outputs": outputs,
            # "neuron_dentrites": neuron_dendrites,
        }

        # TODO

        self.input_weights = np.array([[0, 0],[0, 0]], dtype=Weight)
        self.hidden_weights = np.array([[[0, 0],[0, 0]], [[0, 0],[0, 0]]], dtype=Weight)
        self.output_weights = np.array([[0, 0],[0, 0]], dtype=Weight)

        print(f"Graia model with {self.parameters} parameters ready.")


    def fit(self, xs: NDArray[InputVal], ys: NDArray[OutputVal], epochs: int):
        # xs2 = np.array([[255, 0], [0, 255], [0, 0], [255, 255], [200, 0], [0, 200]], dtype=InputVal)
        # ys2 = np.array([1, 2, 3, 4, 5, 6], dtype=OutputVal)
        return g.fit(
            self.input_weights, self.hidden_weights, self.output_weights,
            xs, ys, np.int32(epochs)
        )
