-- 🌄 Graia: an *experimental* neural network library.

-- Wt = Weight
-- Weights are negative for inhibition, positive for excitation, zero for no connection
type Wt = f32 -- real f32 on GPUs, emulated as f32 on CPUs.

-- Val = Value of a node (input, hidden and output)
type Val = f32

type TeachCfg = {
    learningRate: f32,
    wasGood: bool,
    loss: f32,
    previousLoss: f32
}

-- SKIP ==
-- entry: activation
-- input { 1 -10 } output { 0u8 }
-- input { 2 -10 } output { 0u8 }
-- input { 1 127 } output { 127u8 }
-- input { 2 127 } output { 254u8 }
-- input { 3 127 } output { 255u8 }
def activation (inputsNb: i64) (reluSlope: f32) (s: f32): Val =
    -- ReLU
    if s <= 0 then
        0
    else
        -- (s * reluSlope) / (f32.i64 inputsNb) -- + 0.1
        --  |> f32.min 1
        if reluSlope == 0 then 1 else f32.min 1 ((s * reluSlope) / (f32.i64 inputsNb))

def dotProduct [j] (inputs: [j]Val) (wts: [j]Wt): f32 =
    reduce (+) 0 (map2 (*) inputs wts)

def output [j] (reluSlope: f32) (inputs: [j]Val) (wts: [j]Wt): Val =
    dotProduct inputs wts
    |> activation j reluSlope

-- input layer with j nodes -> output layer with k nodes
def outputs [k] [j] (reluSlope: f32) (inputs: [j]Val) (interWts: [k][j]Wt): [k]Val =
    interWts
    |> map (output reluSlope inputs)

-- changes weights between two layers using last input values
def teachInterLastInputs [k] [j] (reluSlope: f32) (teachCfg: TeachCfg) (interWts: [k][j]Wt) (lastInputs: [j]Val) : [k][j]Wt =
    let { learningRate, wasGood, loss, previousLoss } = teachCfg
    let wasBetter = loss < previousLoss
    in
    interWts
    |> map (\nodeWts ->
        let lastOutput = output reluSlope lastInputs nodeWts
        let wasNodeTriggered = lastOutput > 0
        in
        zip nodeWts lastInputs
        |> map (\(w, lastInput) ->
            let inputContrib = lastInput * w
            let wasInputTriggered = lastInput > 0
            let step = learningRate * loss
            in
            (if wasGood then
                -- Hebbian learning rule
                if wasInputTriggered then
                    if wasNodeTriggered then
                        w + step
                    else
                        w - step
                else
                    w -- * (1 - learningRate)
            else
                if wasInputTriggered then
                    if wasNodeTriggered then
                        w - step
                    else
                       w + step
                else
                    w -- * (1 - learningRate)
            )
            |> f32.min 1.0
            |> f32.max (- 1.0)
        )
    )

def outputsLayers [lmo] [n] (reluSlope: f32) (inputs: [n]Val) (interWtsLayers: [lmo][n][n]Wt): [lmo][n]Val =
    let inputsFill = tabulate_2d (lmo - 1) n (\_ _ -> 0f32)
    in
    foldl (\valsLayers interWts ->
        let vals = outputs reluSlope (last valsLayers) interWts
        in
        (tail valsLayers) ++ [vals] |> sized lmo
    ) (inputsFill ++ [inputs] |> sized lmo) interWtsLayers

-- SKIP ==
-- entry: indexOfGreatest
-- input { [3u8, 8u8, 11u8, 7u8] } output { 2i64 }
-- input { [3u8, 8u8, 11u8, 11u8] } output { 2i64 }
def indexOfGreatest (ys: []Val) : i64 =
    let (_, index) =
        loop (greatestVal, index) = (0, 0) for i < length ys do
            if ys[i] > greatestVal then (ys[i], i) else (greatestVal, index)
    in index

-- SKIP ==
-- entry: getLoss
-- input { [0u8, 0u8, 255u8, 0u8] 2i64 } output { 0u8 }
-- input { [255u8, 255u8, 0u8, 255u8] 2i64 } output { 255u8 }
-- input { [0u8, 0u8, 255u8, 0u8] 1i64 } output { 127u8 }
def getLoss [o] (outputVals: [o]Val) (correctIndex: i64) : f32 =
    let idealOutputVals = tabulate o (\i -> if i == correctIndex then 1 else 0)
    let moxOutput = outputVals[indexOfGreatest outputVals]
    let normalizationCoef = if moxOutput == 0 then 1 else (1 / moxOutput)
    in
    zip outputVals idealOutputVals
    -- mean absolute error
    |> map (\(out, ideal) -> f32.abs ((out * normalizationCoef) - ideal))
    |> reduce (+) 0
    |> (\sum -> sum / f32.i64 o)

-- i = inputs
-- n = nodes per layer
-- o = outputs
-- lmo = layers minus one
-- r = rows
entry fit [r][i][n][lmo][o]
    (inputWts: [n][i]Wt) (hiddenWtsLayers: [lmo][n][n]Wt) (outputWts: [o][n]Wt)
    (learningRate: f32)  (reluSlope: f32)
    (xsRows: [r][i]Val) (yRows: [r]i64)
    : ([n][i]Wt, [lmo][n][n]Wt, [o][n]Wt, i32, f32, i64, [o]Val, [lmo + 1][n]Val, f32) =
    foldl (\(iWts, hWtsLayers, oWts, goodAnswers, totalLoss, _, _, _, previousLoss) (xs, y) ->
        let inputVals = outputs 0 xs iWts
        let hiddenValsLayers = outputsLayers 0 inputVals hWtsLayers
        -- TODO special outputs function for outputVals so reluSlope parameter isn't needed
        let outputVals = outputs reluSlope (last hiddenValsLayers) oWts
        let answer = indexOfGreatest outputVals
        let loss = getLoss outputVals y
        let wasGood = answer == y  && outputVals[answer] > 0
        let teachCfg = { learningRate, wasGood, loss, previousLoss }
        in
        ( teachInterLastInputs 0 teachCfg iWts xs
        , zip hWtsLayers (sized lmo ([inputVals] ++ init hiddenValsLayers))
            |> map (\(wts, ins) -> teachInterLastInputs 0 teachCfg wts ins)
        , teachInterLastInputs 0 teachCfg oWts (last hiddenValsLayers)
        , goodAnswers + if wasGood then 1 else 0
        , totalLoss + loss
        , answer
        , outputVals
        , [inputVals] ++ hiddenValsLayers |> sized (lmo + 1)
        , loss
        )
    )
    (inputWts, hiddenWtsLayers, outputWts, 0i32, 0f32, 0i64, (tabulate o (\_ -> 0f32)), tabulate_2d (lmo + 1) n (\_ _ -> 0f32), 1.0)
    (zip xsRows yRows)

-- TODO
entry predict (x: i32): i32 =
    x + 42
