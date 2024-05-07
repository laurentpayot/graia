from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray
from futhark_ffi import Futhark

from lib import _graia

VERSION = "0.0.1"

graia = Futhark(_graia)
print(f"🌄 Graia v{VERSION}")

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
        boost: int = 64,
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
            "graia_version": VERSION,
            "inputs": inputs,
            "layer_nodes": layer_nodes,
            "layers": layers,
            "outputs": outputs,
            "boost": boost,
            # "node_dentrites": node_inputs,
        }
        self.input_weights = rng.integers(
            -127, 127, size=(layer_nodes, inputs), dtype=np.int8
        )
        self.hidden_weights = rng.integers(
            -127, 127, size=(layers - 1, layer_nodes, layer_nodes), dtype=np.int8
        )
        self.output_weights = rng.integers(
            -127, 127, size=(outputs, layer_nodes), dtype=np.int8
        )
        self.accuracy_history: list[float] = []
        print(f"🌄 Graia model with {self.parameters:,} parameters ready.")

    def fit(
        self,
        xs: NDArray[InputVal],
        ys: NDArray[OutputVal],
        epochs: int,
        learning_divider: np.uint8 = 255,
    ) -> None:
        start = len(self.accuracy_history)
        stop = start + epochs
        for epoch in range(1, epochs + 1):
            (
                input_weights,
                hidden_weights,
                output_weights,
                correct_answers,
                last_answer,
                last_outputs,
                last_intermediate_outputs,
            ) = graia.fit(
                learning_divider,
                self.input_weights,
                self.hidden_weights,
                self.output_weights,
                np.int32(self.config["boost"]),
                xs,
                ys,
            )
            self.input_weights = graia.from_futhark(input_weights)
            self.hidden_weights = graia.from_futhark(hidden_weights)
            self.output_weights = graia.from_futhark(output_weights)
            self.last_answer = last_answer
            self.last_outputs = graia.from_futhark(last_outputs)
            self.last_intermediate_outputs = graia.from_futhark(
                last_intermediate_outputs
            )
            if len(xs) == 1:
                isCorrect = ys[0] == last_answer
                print(f"Epoch {epoch}/{epochs}: answer {last_answer} is {isCorrect}")
            else:
                accuracy = correct_answers / ys.size
                self.accuracy_history.append(accuracy)
                print(f"Epoch {start + epoch}/{stop}: accuracy {100 * accuracy :.3f}%")

    # def teachInput(self) -> None:
    #     g.teachInter(np.int8(1), False, self.input_weights)
