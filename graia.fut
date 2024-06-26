-- 🌄 Graia: an *experimental* neural network library.

-- Wt = Weight
-- A weight of n is actually the inverse of 2 at the power of n (bitwise right shift by abs(n) - 1)
-- Weights are negative for inhibition, positive for excitation, zero for no connection
type Wt = i8

-- Val = Value of a node (input, hidden and output)
type Val = u8

type TeachCfg = {
    maxWt: i8,
    wasGood: bool,
    loss: u8,
    previousLoss: u8
}

-- ==
-- entry: signedRightShift
-- input { 0i8 200u8 } output { 0 }
-- input { 8i8 200u8 } output { 1 }
-- input { -8i8 200u8 } output { -1 }
-- input { 9i8 200u8 } output { 0 }
-- input { -9i8 200u8 } output { 0 }
-- input { 1i8 200u8 } output { 200 }
-- input { 2i8 200u8 } output { 100 }
-- input { 3i8 200u8 } output { 50 }
-- input { -1i8 200u8 } output { -200 }
-- input { -2i8 200u8 } output { -100 }
-- input { -3i8 200u8 } output { -50 }
def signedRightShift (w: Wt) (v: Val): i32 =
    if w == 0 then
        0
    else
        if w > 0 then
            i32.u8 (v >> u8.i8 (w - 1))
        else
            - i32.u8 (v >> u8.i8 (-w - 1))

-- ==
-- entry: activation
-- input { 1 -10 } output { 0u8 }
-- input { 2 -10 } output { 0u8 }
-- input { 1 127 } output { 127u8 }
-- input { 2 127 } output { 254u8 }
-- input { 3 127 } output { 255u8 }
def activation (reluBoost: i32) (s: i32): Val =
    -- ReLU
    if s <= 0 then 0 else u8.i32 <| i32.min 255 <|
        (s * reluBoost)

def dotShift [j] (inputs: [j]Val) (wts: [j]Wt): i32 =
    reduce (+) 0 (map2 signedRightShift wts inputs)

def output [j] (reluBoost: i32) (inputs: [j]Val) (wts: [j]Wt): Val =
    dotShift inputs wts
    |> activation reluBoost

-- input layer with j nodes -> output layer with k nodes
def outputs [k] [j] (reluBoost: i32) (inputs: [j]Val) (interWts: [k][j]Wt): [k]Val =
    interWts
    |> map (output reluBoost inputs)

def exciteFor (maxWt: Wt) (step: i8) (w: Wt): i8 =
    if w == -maxWt then
        maxWt - 1
    else
        if w == 1 then
            1
        else
            w - step

def inhibitFor (maxWt: Wt) (step: i8) (w: Wt): i8 =
    if w == maxWt then
        -maxWt + 1
    else
        if w == -1 then
            -1
        else
            w + step

-- changes weights between two layers using last input values
def teachInterLastInputs [k] [j] (reluBoost: i32) (teachCfg: TeachCfg) (interWts: [k][j]Wt) (lastInputs: [j]Val) : [k][j]Wt =
    let { maxWt, wasGood, loss, previousLoss } = teachCfg
    let wasBetter = loss < previousLoss
    let excite = exciteFor maxWt 1
    let inhibit = inhibitFor maxWt 1
    in
    interWts
    |> map (\nodeWts ->
        let lastOutput = output reluBoost lastInputs nodeWts
        let wasNodeTriggered = lastOutput > 0
        in
        zip nodeWts lastInputs
        |> map (\(w, lastInput) ->
            let wasInputTriggered = lastInput > 0
            -- let inputContrib = signedRightShift w lastInput
            in
            if wasBetter then
                -- Hebbian learning rule
                if wasInputTriggered then
                    if wasNodeTriggered then
                        excite w
                    else
                        inhibit w
                else
                    w
            else
                if wasInputTriggered then
                    if wasNodeTriggered then
                        inhibit w
                    else
                        excite w
                else
                    w
        )
    )

def outputsLayers [lmo] [n] (reluBoost: i32) (inputs: [n]Val) (interWtsLayers: [lmo][n][n]Wt): [lmo][n]Val =
    let inputsFill = tabulate_2d (lmo - 1) n (\_ _ -> 0u8)
    in
    foldl (\valsLayers interWts ->
        let vals = outputs reluBoost (last valsLayers) interWts
        in
        (tail valsLayers) ++ [vals] |> sized lmo
    ) (inputsFill ++ [inputs] |> sized lmo) interWtsLayers

-- ==
-- entry: indexOfGreatest
-- input { [3u8, 8u8, 11u8, 7u8] } output { 2i64 }
-- input { [3u8, 8u8, 11u8, 11u8] } output { 2i64 }
def indexOfGreatest (ys: []u8) : i64 =
    let (_, index) =
        loop (greatestVal, index) = (0, 0) for i < length ys do
            if ys[i] > greatestVal then (ys[i], i) else (greatestVal, index)
    in index

-- ==
-- entry: getLoss
-- input { [0u8, 0u8, 255u8, 0u8] 2i64 } output { 0u8 }
-- input { [255u8, 255u8, 0u8, 255u8] 2i64 } output { 255u8 }
-- input { [0u8, 0u8, 255u8, 0u8] 1i64 } output { 127u8 }
def getLoss [o] (outputVals: [o]Val) (correctIndex: i64) : u8 =
    let idealOutputVals = tabulate o (\i -> if i == correctIndex then 255 else 0)
    in
    zip outputVals idealOutputVals
    -- mean absolute error
    |> map (\(out, ideal) -> i32.abs (i32.u8(out) - ideal))
    |> reduce (+) 0
    |> (\sum -> sum / i32.i64 o)
    |> u8.i32

-- i = inputs
-- n = nodes per layer
-- o = outputs
-- lmo = layers minus one
-- r = rows
entry fit [r][i][n][lmo][o]
    (maxWt: i8) (inputWts: [n][i]Wt) (hiddenWtsLayers: [lmo][n][n]Wt) (outputWts: [o][n]Wt) (reluBoost: i32)
    (xsRows: [r][i]Val) (yRows: [r]Val)
    : ([n][i]Wt, [lmo][n][n]Wt, [o][n]Wt, i32, i32, i64, [o]Val, [lmo + 1][n]Val, u8) =
    foldl (\(iWts, hWtsLayers, oWts, goodAnswers, totalLoss, _, _, _, previousLoss) (xs, y) ->
        let inputVals = outputs reluBoost xs iWts
        let hiddenValsLayers = outputsLayers reluBoost inputVals hWtsLayers
        let outputVals = outputs reluBoost (last hiddenValsLayers) oWts
        let answer = indexOfGreatest outputVals
        let loss = getLoss outputVals (i64.u8 y)
        let wasGood = answer == i64.u8 y && loss < 127
        let teachCfg = { maxWt, wasGood, loss, previousLoss }
        in
        ( teachInterLastInputs reluBoost teachCfg iWts xs
        , zip hWtsLayers (sized lmo ([inputVals] ++ init hiddenValsLayers))
            |> map (\(wts, ins) -> teachInterLastInputs reluBoost teachCfg wts ins)
        , teachInterLastInputs reluBoost teachCfg oWts (last hiddenValsLayers)
        , goodAnswers + if wasGood then 1 else 0
        , totalLoss + i32.u8 loss
        , answer
        , outputVals
        , [inputVals] ++ hiddenValsLayers |> sized (lmo + 1)
        , loss
        )
    )
    (inputWts, hiddenWtsLayers, outputWts, 0, 0, 0, (tabulate o (\_ -> 0u8)), tabulate_2d (lmo + 1) n (\_ _ -> 0u8), 255)
    (zip xsRows yRows)

-- TODO
entry predict (x: i32): i32 =
    x + 42
