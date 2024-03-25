-- Graia

-- Wt = Weight
-- A weight of n is actually the inverse of 2 at the power of n (right shift by abs(n) - 1)
-- Weights are negative for inhibition, positive for excitation, zero for no connection
type Wt = i8

-- Val = Value of a node (input, hidden and output)
type Val = u8

type TeachCfg = {
    maxWt: i8,
    learningStep: i8,
    wasGood: bool
}

-- changes weights between two layers
def teachInter [k] [j] (maxWt: i8) (learningStep: i8) (wasGood: bool) (interWts: [k][j]Wt) : [k][j]Wt =
    interWts
    |> map (\nodeWts ->
        nodeWts
        |> map (\wt ->
            if wt == 0 then
                if wasGood then maxWt else -maxWt
            else
                if wt > 0 then
                    if wasGood then wt - learningStep else wt + learningStep
                else
                    if wasGood then wt + learningStep else wt - learningStep
        )
    )

def changeWt (teachCfg: TeachCfg) (wt: Wt) : Wt =
    let { maxWt, learningStep, wasGood } = teachCfg
    in
    if wt == 0 then
        if wasGood then maxWt else -maxWt
    else
        if wt > 0 then
            if wasGood then wt - learningStep else wt + learningStep
        else
            if wasGood then wt + learningStep else wt - learningStep

-- ==
-- entry: signedRightShift
-- input { 0i8 200u8 } output { 0i16 }
-- input { 1i8 200u8 } output { 100i16 }
-- input { 2i8 200u8 } output { 50i16 }
-- input { -1i8 200u8 } output { -100i16 }
-- input { -2i8 200u8 } output { -50i16 }
def signedRightShift (w: Wt) (v: Val): i16 =
    -- (i16.sgn w) * i16.u8 (v >> u8.i8 (i8.abs w))
    if w == 0 then
        0
    else
        if w > 0 then
           i16.u8 (v >> u8.i8 w)
        else
            - i16.u8 (v >> u8.i8 (-w))

def activation (s: i16): Val =
    -- ReLU
    if s > 0 then u8.i16 (i16.min s 255) else 0

def nodeOps [j] (teachCfg: TeachCfg) (inputs: [j]Val) (inputWts: [j]Wt): ([j]Wt, Val) =
    zip inputWts inputs
    |> map (\(w, v) ->
        let w' = changeWt teachCfg w
        in
        (w', (signedRightShift w' v))
    )
    |> unzip
    |> \(wts, wtVals) -> (wts, reduce (+) 0 wtVals)
    |> \(wts, sum) -> (wts, activation sum)

-- input layer with j nodes -> output layer with k nodes
def outputs [k] [j] (interWts: [k][j]Wt) (inputs: [j]Val): [k]Val =
    interWts
    |> map (\inputWts ->
        loop acc: i16 = 0 for (w, v) in zip inputWts inputs do
            acc + (signedRightShift w v)
    )
    |> map activation

-- input layer with j nodes -> output layer with k nodes
def outputs2 [k] [j] (teachCfg: TeachCfg) (interWts: [k][j]Wt) (inputs: [j]Val): ([k][j]Wt, [k]Val) =
    interWts
    |> map (nodeOps teachCfg inputs)
    |> unzip

-- ==
-- entry: indexOfGreatest
-- input { [3u8, 8u8, 11u8, 7u8] } output { 2i64 }
-- input { [3u8, 8u8, 11u8, 11u8] } output { 2i64 }
let indexOfGreatest (ys: []u8) : i64 =
    let (_, index) =
        loop (greatestVal, index) = (0, 0) for i < length ys do
            if ys[i] > greatestVal then (ys[i], i) else (greatestVal, index)
    in index

-- i = inputs
-- n = nodes per layer
-- o = outputs
-- lmo = layers minus one
-- r = rows
entry fit [r][i][n][lmo][o]
    (maxWt: i8) (inputWts: [n][i]Wt) (hiddenWts: [lmo][n][n]Wt) (outputWts: [o][n]Wt)
    ( xs: [r][i]Val) (ys: [r]Val) (learningStep: i8)
    : ([n][i]Wt, [lmo][n][n]Wt, [o][n]Wt, i32, []Val) =
    loop (iWts, hWts, oWts, goodAnswers, _) = (inputWts, hiddenWts, outputWts, 0, []) for (x, y) in zip xs ys do
        let outputVals =
            (loop inputs = outputs inputWts x for k < lmo do
                outputs hiddenWts[k] inputs)
            |> outputs outputWts
        let wasGood =
            outputVals
            |> indexOfGreatest
            |> (==) (i64.u8 y)
        in
        ( teachInter maxWt learningStep wasGood iWts
        , hWts |> map (\wts -> teachInter maxWt learningStep wasGood wts)
        , teachInter maxWt learningStep wasGood oWts
        , goodAnswers + if wasGood then 1 else 0
        , outputVals
        )

entry fit2 [r][i][n][lmo][o]
    (maxWt: i8) (inputWts: [n][i]Wt) (hiddenWts: *[lmo][n][n]Wt) (outputWts: [o][n]Wt)
    ( xs: [r][i]Val) (ys: [r]Val) (learningStep: i8)
    : ([n][i]Wt, *[lmo][n][n]Wt, [o][n]Wt, i32, []Val) =
    let teachCfg: TeachCfg = { maxWt = maxWt, learningStep = learningStep, wasGood = false }
    in
    (loop (iWts, hWts, oWts, goodAnswers, teachCfg, _) = (inputWts, hiddenWts, outputWts, 0, teachCfg, []) for (x, y) in zip xs ys do
        let (iWts', iVals) = outputs2 teachCfg iWts x
        let (oWts', oVals) =
            (loop (wts, inputs) = (hWts, iVals) for layer < lmo do
                outputs2 teachCfg wts[layer] inputs
                |> (\(layerWts, layerOutputs) ->
                    (hiddenWts with [layer] = layerWts, layerOutputs)
                )
            )
            |> (\(_, outputs) -> outputs)
            |> outputs2 teachCfg oWts
        let wasGood =
            oVals
            |> indexOfGreatest
            |> (==) (i64.u8 y)
        in
        ( iWts'
        , hWts
        , oWts'
        , goodAnswers + if wasGood then 1 else 0
        , { learningStep = teachCfg.learningStep, maxWt = teachCfg.maxWt, wasGood = wasGood }
        , oVals
        )
    )
    |> (\(iWts, hWts, oWts, goodAnswers, _, lastOutputs) -> (iWts, hWts, oWts, goodAnswers, lastOutputs))

-- TODO
entry predict (x: i32): i32 =
    x + 42
