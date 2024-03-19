-- Graia

-- Wt = Weight
-- A weight of n is actually the inverse of 2 at the power of n (right shift by abs(n) - 1)
-- Weights are negative for inhibition, positive for excitation, zero for no connection
type Wt = i8

-- i = inputs
-- n = neurons per layer
-- o = outputs
-- lmo = layers minus one
-- r = rows
type Model [i][n][lmo][o] = {
    inputWts: [n][i]Wt,
    hiddenWts: [lmo][n][n]Wt,
    outputWts: [o][n]Wt
}

-- Val = Value of a neuron (input, hidden and output)
type Val = u8

-- -- Sum = Sum of ponderated values
-- type Sum = i16
-- def sumToVal = u8.i16
-- def maxSum: Sum = 255


def signedRightShift (w: Wt) (v: Val): i8 =
    (i8.sgn w) * i8.u8 (v >> u8.i8 (i8.abs w))


-- def getOutput [j] (ws: [j]Wt) (is: [j]Val) : Val =
--   (zip ws is)
--   |> map (\(w,i) -> w*i)
--   foldl (+) 0

-- def feedForward [i][n][lmo][o]
--   (model: Model [i][n][lmo][o]) (wasGood: bool) (inputs: [i]Val)
--   : (Model [i][n][lmo][o], [o]Val) =
--   -- TODO
--   (model, 0)

def activation (s: i16): Val =
    -- ReLU
    if s > 0 then u8.i16 (i16.min s 255) else 0

-- changes weights between two layers
def teachInter [k][j] (learningStep: i8) (wasGood: bool) (neuronInputWts: *[k][j]Wt) : [k][j]Wt =
    let delta: i8 = if wasGood then -learningStep else learningStep
    in
    loop neuronInputWts for neuron < k do
        loop  neuronInputWts for input < j do
            neuronInputWts with [neuron, input] = neuronInputWts[neuron, input] + delta

-- value layer j -> neuron layer k
def layerOutputs [k][j] (neuronInputWts: [k][j]Wt) (inputVals: [j]Val): [k]Val =
    neuronInputWts
    |> map (\inputWts ->
        loop acc: i16 = 0 for (w, v) in zip inputWts inputVals do
            acc + (i16.i8 (signedRightShift w v))
    )
    |> map activation


entry fit [r][i][n][lmo][o]
    (inputWts: *[n][i]Wt) (hiddenWts: [lmo][n][n]Wt) (outputWts: [o][n]Wt)
    ( xs: [r][i]Val) (ys: [r]Val) (learningStep: i8)
    : ([n][i]Wt, [lmo][n][n]Wt, [o][n]Wt, f16) =
    -- TODO
    let model: Model [i][n][lmo][o] = {
        inputWts = inputWts,
        hiddenWts = hiddenWts,
        outputWts =outputWts
    }
    let inputWts' = inputWts with [0, 0] = 42
    in
    -- zip xs ys |> map (\x y -> feedForward model true x)


    (inputWts', hiddenWts, outputWts, 0.0)

    entry predict (x: i32): i32 =
        x + 42
