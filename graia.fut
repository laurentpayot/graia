-- Graia

-- Wt = Weight
-- A weight of n is actually the inverse of 2 at the power of n (right shift by abs(n) - 1)
-- Weights are negative for inhibition, positive for excitation, zero for no connection
type Wt = i8

type InputVal = u8 -- for xs
type AnswerVal = u8 -- for ys

-- Val = output value of a node
type Val = u32

type TeachCfg = {
    maxWt: i8,
    wasGood: bool,
    loss: u32
}


-- ==
-- entry: signedRightShift
-- input { 8i8 200u8 } output { 0 }
-- input { -8i8 200u8 } output { 0 }
-- input { 1i8 200u8 } output { 200 }
-- input { 2i8 200u8 } output { 100 }
-- input { -1i8 200u8 } output { -200 }
-- input { -2i8 200u8 } output { -100 }
def signedRightShift (w: Wt) (v: Val): i32 =
    if i8.abs w == 8 then -- TODO remove hardcoded 8
        0
    else
        if w > 0 then
            i32.u32 (v >> u32.i8 (w - 1))
        else
            -- if needed -1 + 1 (nothing) for less inhibition
            - i32.u32 (v >> u32.i8 (-w - 1))

-- SKIP ==
-- entry: activation
-- input { 2 4i64 127 } output { 63u8 }
-- input { 8 4i64 127 } output { 254u8 }
-- input { 16 4i64 127 } output { 255u32 }
def activation (boost: i32) (s: i32): Val =
    -- ReLU
    if s <= 0 then
        0
    else
        u32.i32 (boost * s)

def dotShift [j] (inputs: [j]Val) (wts: [j]Wt): i32 =
    reduce (+) 0 (map2 signedRightShift wts inputs)

def output [j] (boost: i32) (inputs: [j]Val) (wts: [j]Wt): Val =
    dotShift inputs wts
    |> activation boost

-- input layer with j nodes -> output layer with k nodes
def outputs [k] [j] (boost: i32) (inputs: [j]Val) (interWts: [k][j]Wt): [k]Val =
    interWts
    |> map (output boost inputs)

-- def getStep (maxWt: Wt) (loss: u32) (contrib: i32): i8 =
--     i8.i32 <| ((i32.i8 maxWt - 1) * contrib) / 255

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
def teachInterLastInputs [k] [j] (boost: i32) (teachCfg: TeachCfg) (interWts: [k][j]Wt) (lastInputs: [j]Val) : [k][j]Wt =
    let { maxWt, wasGood, loss } = teachCfg
    let step = 1
    let excite = exciteFor maxWt step
    let inhibit = inhibitFor maxWt step
    -- let lastOutput = outputs 64 lastInputs interWts
    -- let wasGood = loss < 16
    in
    interWts
    |> map (\nodeWts ->
        let lastOutput = output boost lastInputs nodeWts
        let wasTriggered = lastOutput > 0
        in
        zip nodeWts lastInputs
        |> map (\(w, lastInput) ->
            let contrib = signedRightShift w lastInput
            let isToChange = i32.abs contrib < i32.u32 loss
            -- let step = getStep maxWt loss contrib
            in
            if wasTriggered then
            -- if true then
                if isToChange then
                        if wasGood then
                            excite w
                        else
                            inhibit w
                else
                    w
            else
                excite w
        )
    )

def outputsLayers [lmo] [n] (boost: i32) (inputs: [n]Val) (interWtsLayers: [lmo][n][n]Wt): [lmo][n]Val =
    let inputsFill = tabulate_2d (lmo - 1) n (\_ _ -> 0u32)
    in
    foldl (\valsLayers interWts ->
        let vals = outputs boost (last valsLayers) interWts
        in
        (tail valsLayers) ++ [vals] |> sized lmo
    ) (inputsFill ++ [inputs] |> sized lmo) interWtsLayers

-- def inputsLayers [lmo] [n] (boost: i32) (inputs: [n]Val) (interWtsLayers: [lmo][n][n]Wt): [lmo][n]Val =
--     [inputs] ++ outputsLayers boost inputs (init interWtsLayers) |> sized lmo

-- ==
-- entry: indexOfGreatest
-- input { [3u32, 8u32, 11u32, 7u32] } output { 2i64 }
-- input { [3u32, 8u32, 11u32, 11u32] } output { 2i64 }
def indexOfGreatest (ys: []Val) : i64 =
    let (_, index) =
        loop (greatestVal, index) = (0, 0) for i < length ys do
            if ys[i] > greatestVal then (ys[i], i) else (greatestVal, index)
    in index

def greatestVal (ys: []Val) : Val =
    loop greatest = 0 for i < length ys do
        if ys[i] > greatest then ys[i] else greatest

-- TODO ==
-- entry: getLoss
-- input { [0u32, 0u32, 255u32, 0u32] 2i64 } output { 0u32 }
-- input { [255u32, 255u32, 0u32, 255u32] 2i64 } output { 255u32 }
-- input { [0u32, 0u32, 255u32, 0u32] 1i64 } output { 127u32 }
def getLoss [o] (outputVals: [o]Val) (correctIndex: i64) : u32 =
    let greatestOutputVal = greatestVal outputVals
    let idealOutputVals = tabulate o (\i -> if i == correctIndex then greatestOutputVal else 0)
    in
    zip outputVals idealOutputVals
    -- mean absolute error
    |> map (\(out, ideal) -> i64.abs (i64.u32 out - i64.u32 ideal))
    |> reduce (+) 0
    |> (\sum -> sum / o)
    |> u32.i64

-- i = inputs
-- n = nodes per layer
-- o = outputs
-- lmo = layers minus one
-- r = rows
entry fit [r][i][n][lmo][o]
    (maxWt: i8) (inputWts: [n][i]Wt) (hiddenWtsLayers: [lmo][n][n]Wt) (outputWts: [o][n]Wt) (boost: i32)
    (xsRows: [r][i]InputVal) (yRows: [r]AnswerVal)
    : ([n][i]Wt, [lmo][n][n]Wt, [o][n]Wt, i32, [o]Val, [lmo + 1][n]Val) =
    foldl (\(iWts, hWtsLayers, oWts, goodAnswers, _, _) (xs', y) ->
        let xs = map u32.u8 xs'
        let inputVals = outputs boost xs iWts
        let hiddenValsLayers = outputsLayers boost inputVals hWtsLayers
        let outputVals = outputs boost (last hiddenValsLayers) oWts
        let wasGood =
            outputVals
            |> indexOfGreatest
            |> (==) (i64.u8 y)
        let loss = getLoss outputVals (i64.u8 y)
        let teachCfg = { maxWt, wasGood, loss }
        in
        ( teachInterLastInputs boost teachCfg iWts xs
        , zip hWtsLayers (sized lmo ([inputVals] ++ init hiddenValsLayers))
            |> map (\(wts, ins) -> teachInterLastInputs boost teachCfg wts ins)
        , teachInterLastInputs boost teachCfg oWts (last hiddenValsLayers)
        , goodAnswers + if wasGood then 1 else 0
        , outputVals
        , [inputVals] ++ hiddenValsLayers |> sized (lmo + 1)
        )
    )
    (inputWts, hiddenWtsLayers, outputWts, 0, (tabulate o (\_ -> 0u32)), tabulate_2d (lmo + 1) n (\_ _ -> 0u32))
    (zip xsRows yRows)

-- TODO
entry predict (x: i32): i32 =
    x + 42
