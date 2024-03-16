from typing import TypeAlias, TypedDict
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


Weights: TypeAlias = NDArray[np.int8]
Inputs: TypeAlias = NDArray[np.uint8]
Outputs: TypeAlias = NDArray[np.uint8]

class ModelWeights(TypedDict):
    input: Weights
    hidden: Weights
    output: Weights

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
        # A weight of n is actually the inverse of 2 at the power of n (right shift by abs(n) - 1)
        # Weights are negative for inhibition, positive for excitation, zero for no connection
        self.weights : ModelWeights = {
            "input": np.array([[0],[0]], dtype=np.int8),
            "hidden": np.array([[0],[0]], dtype=np.int8),
            "output": np.array([[0],[0]], dtype=np.int8),
        }
        print(f"Graia model with {self.parameters} parameters ready.")


    def fit(self, xs: Inputs, ys: Outputs, epochs: int):
        return g.fit(xs, ys, np.int32(epochs))
