-- Graia

-- Wt = Weight
-- A weight of n is actually the inverse of 2 at the power of n (right shift by abs(n) - 1)
-- Weights are negative for inhibition, positive for excitation, zero for no connection
type Wt = i8

-- i = inputs
-- n = nodes per layer
-- o = outputs
-- lmo = layers minus one
-- r = rows
type Model [i][n][lmo][o] = {
    inputWts: [n][i]Wt,
    hiddenWts: [lmo][n][n]Wt,
    outputWts: [o][n]Wt
}

-- Val = Value of a node (input, hidden and output)
type Val = u8

-- -- Sum = Sum of ponderated values
-- type Sum = i16
-- def sumToVal = u8.i16
-- def maxSum: Sum = 255


def signedRightShift (w: Wt) (v: Val): i8 =
    (i8.sgn w) * i8.u8 (v >> u8.i8 (i8.abs w))

def activation (s: i16): Val =
    -- ReLU
    if s > 0 then u8.i16 (i16.min s 255) else 0

-- changes weights between two layers
def teachInter [k] [j] (maxWt: i8) (learningStep: i8) (wasGood: bool) (interWts: *[k][j]Wt) : [k][j]Wt =
    loop interWts for node < k do
        loop interWts for input < j do
            let wt = interWts[node, input]
            in
            interWts with [node, input] =
                if wt == 0 then
                    if wasGood then maxWt else -maxWt
                else
                    if wt > 0 then
                        if wasGood then wt - learningStep else wt + learningStep
                    else
                        if wasGood then wt + learningStep else wt - learningStep


-- input layer with j nodes -> output layer with k nodes
def outputs [k] [j] (interWts: [k][j]Wt) (inputs: [j]Val): [k]Val =
    interWts
    |> map (\inputWts ->
        loop acc: i16 = 0 for (w, v) in zip inputWts inputs do
            acc + (i16.i8 (signedRightShift w v))
    )
    |> map activation

-- ==
-- entry: indexOfGreatest
-- input { [3u8, 8u8, 11u8, 7u8] } output { 2i64 }
let indexOfGreatest (ys: []u8) : i64 =
    let (_, index) =
        loop (greatestVal, index) = (0, 0) for i < length ys do
            if ys[i] > greatestVal then (ys[i], i) else (greatestVal, index)
    in index

def isGood [i][n][lmo][o]
    (inputWts: [n][i]Wt) (hiddenWts: [lmo][n][n]Wt) (outputWts: [o][n]Wt)
    (x: [i]Val) (y: Val): bool =
    (loop inputs = outputs inputWts x for k < lmo do
        outputs hiddenWts[k] inputs)
    |> outputs outputWts
    |> indexOfGreatest
    |> (==) (i64.u8 y)

entry fit [r][i][n][lmo][o]
    (maxWt: i8) (inputWts: [n][i]Wt) (hiddenWts: [lmo][n][n]Wt) (outputWts: [o][n]Wt)
    ( xs: [r][i]Val) (ys: [r]Val) (learningStep: i8)
    : ([n][i]Wt, [lmo][n][n]Wt, [o][n]Wt, f16) =
    -- let model: Model [i][n][lmo][o] = {
    --     inputWts = inputWts,
    --     hiddenWts = hiddenWts,
    --     outputWts =outputWts
    -- }
    let teachInput [n] [i] (wasGood: bool) (interWts: *[n][i]Wt) = teachInter maxWt learningStep
    let teachHidden [n] (wasGood: bool) (interWts: *[n][n]Wt) = teachInter maxWt learningStep
    let teachOutput [o] [n] (wasGood: bool) (interWts: *[o][n]Wt) = teachInter maxWt learningStep
    in


    (inputWts, hiddenWts, outputWts, 0.0)

entry predict (x: i32): i32 =
    x + 42
