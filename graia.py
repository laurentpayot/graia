from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray
from futhark_ffi import Futhark

from lib import _graia

VERSION = "0.0.1"

print(f"\n🌄 Graia v{VERSION}\n")
print("Graia initializing…")
graia = Futhark(_graia)
print("Graia ready.\n")

# A weight of n is actually the division by 2 at the power of n (right shift by abs(n))
# Weights are negative for inhibition, positive for excitation, zero for no connection
Weight: TypeAlias = np.int8

InputVal: TypeAlias = np.uint8
OutputVal: TypeAlias = np.uint8


class Graia:
    def __init__(
        self,
        inputs: int,
        layer_nodes: int,
        layers: int,
        outputs: int,
        max_weight: int = 6,  # maximum 6 for unsigned 8 bit integers
        # TODO
        # node_inputs=0,
    ) -> None:

        self.parameters: int = (
            (inputs * layer_nodes)
            + (layer_nodes * layer_nodes * (layers - 1))
            + (layer_nodes * outputs)
        )

        self.config: dict = {
            "version": VERSION,
            "inputs": inputs,
            "layer_nodes": layer_nodes,
            "layers": layers,
            "outputs": outputs,
            "max_weight": max_weight,
            # "node_dentrites": node_inputs,
        }

        rng = np.random.default_rng()

        self.input_weights = rng.integers(
            low=-max_weight,
            high=max_weight,
            size=(layer_nodes, inputs),
            dtype=Weight,
            endpoint=True,
        )
        self.hidden_weights = rng.integers(
            low=-max_weight,
            high=max_weight,
            size=(layers - 1, layer_nodes, layer_nodes),
            dtype=Weight,
            endpoint=True,
        )
        self.output_weights = rng.integers(
            low=-max_weight,
            high=max_weight,
            size=(outputs, layer_nodes),
            dtype=Weight,
            endpoint=True,
        )
        self.last_outputs = np.zeros((outputs), dtype=np.uint8)
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
        for epoch in range(1, epochs + 1):
            input_weights, hidden_weights, output_weights, correct, last_outputs = (
                graia.fit(
                    np.int8(self.config["max_weight"]),
                    self.input_weights,
                    self.hidden_weights,
                    self.output_weights,
                    xs,
                    ys,
                    np.int8(learning_step),
                )
            )
            self.input_weights = graia.from_futhark(input_weights)
            self.hidden_weights = graia.from_futhark(hidden_weights)
            self.output_weights = graia.from_futhark(output_weights)
            self.last_outputs = graia.from_futhark(last_outputs)
            print(f"Epoch {epoch}/{epochs}: correct = {correct}")

    # def teachInput(self) -> None:
    #     g.teachInter(np.int8(1), False, self.input_weights)
