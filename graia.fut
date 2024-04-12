-- Graia

-- Wt = Weight
-- A weight of n is actually the inverse of 2 at the power of n (right shift by abs(n) - 1)
-- Weights are negative for inhibition, positive for excitation, zero for no connection
type Wt = i8

-- Val = Value of a node (input, hidden and output)
type Val = u8

type TeachCfg = {
    maxWt: i8,
    wasGood: bool,
    loss: u8
}

-- ==
-- entry: getStep
-- input { 8i8 255u8 255u8 } output { 7i8 }
-- input { 8i8 255u8 127u8 } output { 3i8 }
-- input { 8i8 0u8 255u8 } output { 0i8 }
-- input { 8i8 127u8 255u8 } output { 3i8 }
def getStep (maxWt: Wt) (loss: u8) (lastOutput: Val): i8 =
    i8.i32 <| ((i32.i8 maxWt - 1) * (i32.u8 loss) * (i32.u8 lastOutput)) / 65025i32

-- changes weights between two layers
def teachInter [k] [j] (teachCfg: TeachCfg) (interWts: [k][j]Wt) (lastOutputs: [k]Val) : [k][j]Wt =
    let { maxWt, wasGood, loss } = teachCfg
    -- let step = 1 --(getStep maxWt loss)
    -- let wasGood = loss < 16
    in
    zip interWts lastOutputs
    |> map (\(nodeWts, lastOutput) ->
        let wasTriggered = lastOutput > 0
        in
        nodeWts
        |> map (\wt ->
            let isToChange = 2 ** (maxWt - i8.abs wt) < i8.u8 loss
            in
            if isToChange then
                let step = getStep maxWt loss lastOutput
                -- let step = 1
                in
                -- if wt > 0 then
                --     if not isToChange then i8.min 1 (wt + step) else i8.min maxWt (wt + step)
                -- else
                --     if not isToChange then i8.max (-1) (wt + step) else i8.max (-maxWt) (wt - step)
                if wt > 0 then
                    if wasTriggered then i8.min maxWt (wt + step) else i8.min 1 (wt - step)
                else
                    if wasTriggered then i8.max (-1) (wt + step) else i8.max (-maxWt) (wt - step)
            else
                wt
        )
    )

-- SKIP ==
-- entry: signedRightShift
-- input { 0i8 200u8 } output { 0 }
-- input { 1i8 200u8 } output { 100 }
-- input { 2i8 200u8 } output { 50 }
-- input { -1i8 200u8 } output { -100 }
-- input { -2i8 200u8 } output { -50 }
def signedRightShift (w: Wt) (v: Val): i32 =
    if w > 0 then
        i32.u8 (v >> u8.i8 w)
    else
        - i32.u8 (v >> u8.i8 (-w))

-- ==
-- entry: activation
-- input { 2 2i64 127 } output { 127u8 }
-- input { 2 4i64 127 } output { 63u8 }
-- input { 8 4i64 127 } output { 254u8 }
-- input { 16 4i64 127 } output { 255u8 }
def activation (boost: i32) (inputs: i64) (s: i32): Val =
    -- ReLU
    if s <= 0 then 0 else u8.i32 <| i32.min 255 <|
        (boost * s) / (i32.i64 inputs)

def dotShift [j] (inputs: [j]Val) (wts: [j]Wt): i32 =
    reduce (+) 0 (map2 signedRightShift wts inputs)

-- input layer with j nodes -> output layer with k nodes
def outputs [k] [j] (boost: i32) (inputs: [j]Val) (interWts: [k][j]Wt): [k]Val =
    interWts
    |> map (dotShift inputs)
    |> map (activation boost j)

def teachInter2 [k] [j] (teachCfg: TeachCfg) (interWts: [k][j]Wt) (lastInputs: [j]Val) : [k][j]Wt =
    let { maxWt, wasGood, loss } = teachCfg
    -- let lastOutput = outputs 64 lastInputs interWts
    -- let wasGood = loss < 16
    in
    interWts
    |> map (\nodeWts ->
        let lastOutput = dotShift lastInputs nodeWts |> activation 64 j
        in
        zip nodeWts lastInputs
        |> map (\(wt, lastInput) ->
            let contrib = signedRightShift wt lastInput
            let wasBig =  i32.abs contrib < i32.u8 loss
            let step = 1
            in
            if wt > 0 then
                if wasBig then i8.min maxWt (wt + step) else i8.min 1 (wt - step)
            else
                if wasBig then i8.max (-1) (wt + step) else i8.max (-maxWt) (wt - step)
        )
    )

def outputsLayers [lmo] [n] (boost: i32) (inputs: [n]Val) (interWtsLayers: [lmo][n][n]Wt): [lmo][n]Val =
    let inputsFill = tabulate_2d (lmo - 1) n (\_ _ -> 0u8)
    in
    foldl (\valsLayers interWts ->
        let vals = outputs boost (last valsLayers) interWts
        in
        (tail valsLayers) ++ [vals] |> sized lmo
    ) (inputsFill ++ [inputs] |> sized lmo) interWtsLayers

def inputsLayers [lmo] [n] (boost: i32) (inputs: [n]Val) (interWtsLayers: [lmo][n][n]Wt): [lmo][n]Val =
    [inputs] ++ outputsLayers boost inputs (init interWtsLayers) |> sized lmo

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
    (maxWt: i8) (inputWts: [n][i]Wt) (hiddenWtsLayers: [lmo][n][n]Wt) (outputWts: [o][n]Wt) (boost: i32)
    (xs: [r][i]Val) (ys: [r]Val)
    : ([n][i]Wt, [lmo][n][n]Wt, [o][n]Wt, i32, [o]Val, [lmo + 1][n]Val) =
    foldl (\(iWts, hWtsLayers, oWts, goodAnswers, _, _) (x, y) ->
        let inputVals = outputs boost x iWts
        let hiddenValsLayers = outputsLayers boost inputVals hWtsLayers
        let outputVals = outputs boost (last hiddenValsLayers) oWts
        let wasGood =
            outputVals
            |> indexOfGreatest
            |> (==) (i64.u8 y)
        let loss = getLoss outputVals (i64.u8 y)
        let teachCfg = { maxWt, wasGood, loss }
        in
        ( teachInter2 teachCfg iWts x
        , zip hWtsLayers (sized lmo ([inputVals] ++ init hiddenValsLayers))
            |> map (\(wts, ins) -> teachInter2 teachCfg wts ins)
        , teachInter2 teachCfg oWts (last hiddenValsLayers)
        , goodAnswers + if wasGood then 1 else 0
        , outputVals
        , [inputVals] ++ hiddenValsLayers |> sized (lmo + 1)
        )
    )
    (inputWts, hiddenWtsLayers, outputWts, 0, (tabulate o (\_ -> 0u8)), tabulate_2d (lmo + 1) n (\_ _ -> 0u8))
    (zip xs ys)

-- TODO
entry predict (x: i32): i32 =
    x + 42
