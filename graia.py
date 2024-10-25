from typing import TypeAlias, TypedDict
import numpy as np
from numpy.typing import NDArray
from futhark_ffi import Futhark

from lib import _graia

VERSION = "0.0.1"

graia = Futhark(_graia)
print(f"🌄 Graia v{VERSION}")

# A weight of n is actually the division by 2 at the power of n (right shift by abs(n))
# Weights are negative for inhibition, positive for excitation, zero for no connection
Weight: TypeAlias = np.int8

InputVal: TypeAlias = np.uint8
OutputVal: TypeAlias = np.uint8


class History(TypedDict):
    accuracy: list[float]
    loss: list[float]


class Graia:
    def __init__(
        self,
        inputs: int,
        layer_nodes: int,
        layers: int,
        outputs: int,
        learning_rate: float = 0.1,
        relu_boost: int = 64,
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
            "learning_rate": learning_rate,
            "relu_boost": relu_boost,
            # "node_dentrites": node_inputs,
        }
        self.input_weights = rng.integers(
            -127, 128, dtype=Weight, size=(layer_nodes, inputs)
        )
        self.hidden_weights = rng.integers(
            -127, 128, dtype=Weight, size=(layers - 1, layer_nodes, layer_nodes)
        )
        self.output_weights = rng.integers(
            -127, 128, dtype=Weight, size=(outputs, layer_nodes)
        )
        self.history: History = {"accuracy": [], "loss": []}
        print(f"🌄 Graia model with {self.parameters:,} parameters ready.")

    def fit(
        self,
        xs: NDArray[InputVal],
        ys: NDArray[OutputVal],
        epochs: int,
    ) -> None:
        start = len(self.history["loss"])
        stop = start + epochs
        for epoch in range(1, epochs + 1):
            (
                input_weights,
                hidden_weights,
                output_weights,
                correct_answers,
                total_loss,
                last_answer,
                last_outputs,
                last_intermediate_outputs,
                previous_loss,
            ) = graia.fit(
                self.input_weights,
                self.hidden_weights,
                self.output_weights,
                np.float32(self.config["learning_rate"]),
                np.int32(self.config["relu_boost"]),
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
            accuracy = correct_answers / ys.size
            self.history["accuracy"].append(accuracy)
            loss = total_loss / (ys.size * 255)
            self.history["loss"].append(loss)
            progress = "█" * (12 * epoch // epochs)
            rest = "░" * (12 - len(progress))
            progress_bar = progress + rest
            if len(xs) == 1:
                print(
                    f"Epoch {epoch}/{epochs}\t {progress_bar}\t answer {last_answer} is {ys[0] == last_answer}",
                    end="\t\r",
                )
            else:
                print(
                    f"Epoch {start + epoch}/{stop}\t {progress_bar}\t Accuracy {100 * accuracy :.3f}%\t Loss (MAE) {100 * loss :.3f}%",
                    end="\t\r",
                )

    # def teachInput(self) -> None:
    #     g.teachInter(np.int8(1), False, self.input_weights)
