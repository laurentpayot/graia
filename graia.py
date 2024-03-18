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
    def __init__(
        self,
        inputs: int,
        layer_neurons: int,
        layers: int,
        outputs: int,
        weight_range: int = 6,  # maximum 6
        # TODO
        # neuron_dendrites=0,
    ) -> None:

        self.parameters: int = (
            (inputs * layer_neurons)
            + (layer_neurons * layer_neurons * (layers - 1))
            + (layer_neurons * outputs)
        )

        self.config: dict = {
            "inputs": inputs,
            "layer_neurons": layer_neurons,
            "layers": layers,
            "outputs": outputs,
            "weight_range": weight_range,
            # "neuron_dentrites": neuron_dendrites,
        }

        rng = np.random.default_rng()

        self.input_weights = rng.integers(
            low=-weight_range,
            high=weight_range,
            size=(layer_neurons, inputs),
            dtype=Weight,
            endpoint=True,
        )
        self.hidden_weights = rng.integers(
            low=-weight_range,
            high=weight_range,
            size=(layers - 1, layer_neurons, layer_neurons),
            dtype=Weight,
            endpoint=True,
        )
        self.output_weights = rng.integers(
            low=-weight_range,
            high=weight_range,
            size=(outputs, layer_neurons),
            dtype=Weight,
            endpoint=True,
        )
        print(
            self.input_weights.shape,
            " -> ",
            self.hidden_weights.shape,
            " -> ",
            self.output_weights.shape,
        )
        print(f"Graia model with {self.parameters} random parameters instantiated.")

    def fit(
        self,
        xs: NDArray[InputVal],
        ys: NDArray[OutputVal],
        epochs: int,
        learning_step=1,  # step of shift changes
    ) -> None:
        for epoch in range(1, epochs):
            self.input_weights, self.hidden_weights, self.output_weights, precision = (
                g.fit(
                    self.input_weights,
                    self.hidden_weights,
                    self.output_weights,
                    xs,
                    ys,
                    np.uint8(learning_step),
                )
            )
            print(f"Epoch {epoch}/{epochs}: precision = {precision}")
