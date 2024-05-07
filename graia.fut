-- Graia

-- Wt = Weight
-- Weights are negative for inhibition, positive for excitation, zero for no connection
type Wt = i8

-- Val = Value of a node (input, hidden and output)
type Val = u8

type TeachCfg = {
    learningDivider: i32,
    wasGood: bool,
    loss: u8
}

def prod (w: Wt) (v: Val): i32 =
    i32.i8 w * i32.u8 v

-- SKIP ==
-- entry: activation
-- input { 2 2i64 127 } output { 127u8 }
-- input { 2 4i64 127 } output { 63u8 }
-- input { 8 4i64 127 } output { 254u8 }
-- input { 16 4i64 127 } output { 255u8 }
def activation (boost: i32) (inputSize: i64) (s: i32): Val =
    -- ReLU
    if s <= 0 then 0 else u8.i32 <| i32.min 255 <|
        (s * boost) --/ (i32.i64 inputSize)

def dotProd [j] (inputs: [j]Val) (wts: [j]Wt): i32 =
    (reduce (+) 0 (map2 prod wts inputs)) / 127

def output [j] (boost: i32) (inputs: [j]Val) (wts: [j]Wt): Val =
    dotProd inputs wts
    |> activation boost j

-- input layer with j nodes -> output layer with k nodes
def outputs [k] [j] (boost: i32) (inputs: [j]Val) (interWts: [k][j]Wt): [k]Val =
    interWts
    |> map (output boost inputs)

-- changes weights between two layers using last input values
def teachInterLastInputs [k] [j] (boost: i32) (teachCfg: TeachCfg) (interWts: [k][j]Wt) (lastInputs: [j]Val) : [k][j]Wt =
    let { learningDivider, wasGood, loss } = teachCfg
    -- let lastOutput = outputs 64 lastInputs interWts
    -- let wasGood = loss < 16
    in
    interWts
    |> map (\nodeWts ->
        let lastOutput = output boost lastInputs nodeWts
        -- let wasTriggered = lastOutput > 0
        in
        zip nodeWts lastInputs
        |> map (\(w, lastInput) ->
            -- let wasInputTriggered = lastInput > 0
            -- let contrib = (prod w lastInput) / 127
            -- let isToChange = i32.abs contrib < i32.u8 loss

            -- Oja’s rule
            let delta =
                (i32.u8 lastOutput * (i32.u8 lastInput - ((prod w lastOutput) / 127)))
                / learningDivider


            -- Laurent Payot’s empiric rule
            -- let delta = contrib - (signedRightShift w lastOutput)
            -- let delta = (signedRightShift w lastOutput) - contrib
            -- let delta = contrib - i32.u8 lastOutput
            -- let delta =  i32.u8 loss - contrib
            -- let delta = contrib - (signedRightShift w lastOutput) + i32.u8 loss
            -- let delta = i32.i8 (w) * (i32.u8 lastInput - (signedRightShift w lastOutput))
            -- let delta = (i32.u8 loss - contrib) * (i32.i8 w)
            -- let delta = -(signedRightShift w loss - signedRightShift w lastOutput)
            -- let delta = -(signedRightShift w loss - contrib)

            in
            if wasGood then
                -- (if delta > 0 then i32.i8 w + 1 else i32.i8 w - 1)
                (i32.i8 w + delta)
                |> i32.max (-127)
                |> i32.min 127
                |> i8.i32
            else
                w
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

-- def inputsLayers [lmo] [n] (boost: i32) (inputs: [n]Val) (interWtsLayers: [lmo][n][n]Wt): [lmo][n]Val =
--     [inputs] ++ outputsLayers boost inputs (init interWtsLayers) |> sized lmo

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
    (learningDivider: i32) (inputWts: [n][i]Wt) (hiddenWtsLayers: [lmo][n][n]Wt) (outputWts: [o][n]Wt) (boost: i32)
    (xsRows: [r][i]Val) (yRows: [r]Val)
    : ([n][i]Wt, [lmo][n][n]Wt, [o][n]Wt, i32, i64, [o]Val, [lmo + 1][n]Val) =
    foldl (\(iWts, hWtsLayers, oWts, goodAnswers, _, _, _) (xs, y) ->
        let inputVals = outputs boost xs iWts
        let hiddenValsLayers = outputsLayers boost inputVals hWtsLayers
        let outputVals = outputs boost (last hiddenValsLayers) oWts
        let answer = indexOfGreatest outputVals
        let loss = getLoss outputVals (i64.u8 y)
        let wasGood = answer == i64.u8 y && loss < 127
        let teachCfg = { learningDivider, wasGood, loss }
        in
        ( teachInterLastInputs boost teachCfg iWts xs
        , zip hWtsLayers (sized lmo ([inputVals] ++ init hiddenValsLayers))
            |> map (\(wts, ins) -> teachInterLastInputs boost teachCfg wts ins)
        , teachInterLastInputs boost teachCfg oWts (last hiddenValsLayers)
        , goodAnswers + if wasGood then 1 else 0
        , answer
        , outputVals
        , [inputVals] ++ hiddenValsLayers |> sized (lmo + 1)
        )
    )
    (inputWts, hiddenWtsLayers, outputWts, 0, 0, (tabulate o (\_ -> 0u8)), tabulate_2d (lmo + 1) n (\_ _ -> 0u8))
    (zip xsRows yRows)

-- TODO
entry predict (x: i32): i32 =
    x + 42
