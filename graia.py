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
        max_weight: int = 8,  # maximum 8 for unsigned 8 bit integers
        boost: int = 20,
        seed: int = None,
        # TODO
        # node_inputs=0,
    ) -> None:

        rng = np.random.default_rng(seed)

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
            "boost": boost,
            # "node_dentrites": node_inputs,
        }
        # no zero weights
        wtsRange = np.concatenate(
            (np.arange(-max_weight, 0), np.arange(1, (max_weight + 1))), dtype=Weight
        )
        # print("Weights range:", wtsRange)
        self.input_weights = rng.choice(wtsRange, size=(layer_nodes, inputs))
        self.hidden_weights = rng.choice(
            wtsRange, size=(layers - 1, layer_nodes, layer_nodes)
        )
        self.output_weights = rng.choice(wtsRange, size=(outputs, layer_nodes))
        self.last_outputs = np.zeros((outputs), dtype=np.uint8)
        print(f"Graia model with {self.parameters} random parameters instantiated.")

    def fit(
        self,
        xs: NDArray[InputVal],
        ys: NDArray[OutputVal],
        epochs: int,
    ) -> None:
        for epoch in range(1, epochs + 1):
            input_weights, hidden_weights, output_weights, correct, last_outputs = (
                graia.fit(
                    np.int8(self.config["max_weight"]),
                    self.input_weights,
                    self.hidden_weights,
                    self.output_weights,
                    np.int32(self.config["boost"]),
                    xs,
                    ys,
                )
            )
            self.input_weights = graia.from_futhark(input_weights)
            self.hidden_weights = graia.from_futhark(hidden_weights)
            self.output_weights = graia.from_futhark(output_weights)
            self.last_outputs = graia.from_futhark(last_outputs)
            accuracy = correct / ys.size
            print(f"Epoch {epoch}/{epochs}: accuracy {100 * accuracy :.3f}%")

    # def teachInput(self) -> None:
    #     g.teachInter(np.int8(1), False, self.input_weights)
