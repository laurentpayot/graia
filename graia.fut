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
def teachInter [k] [j] (teachCfg: TeachCfg) (interWts: [k][j]Wt) : [k][j]Wt =
    let { maxWt, learningStep, wasGood } = teachCfg
    in
    interWts
    |> map (\nodeWts ->
        nodeWts
        |> map (\wt ->
            if wt == maxWt then
                if wasGood then maxWt - learningStep else -maxWt + learningStep
            else if wt == -maxWt then
                if wasGood then -maxWt + learningStep else maxWt - learningStep
            else if wt == 1 then
                if wasGood then 1 else 1 + learningStep
            else if wt == -1 then
                if wasGood then -1 else -1 - learningStep
            else if wt > 0 then
                if wasGood then wt - learningStep else wt + learningStep
            else
                if wasGood then wt + learningStep else wt - learningStep
        )
    )

-- SKIP ==
-- entry: signedRightShift
-- input { 0i8 200u8 } output { 0i16 }
-- input { 1i8 200u8 } output { 100i16 }
-- input { 2i8 200u8 } output { 50i16 }
-- input { -1i8 200u8 } output { -100i16 }
-- input { -2i8 200u8 } output { -50i16 }
def signedRightShift (w: Wt) (v: Val): i16 =
    -- (i16.sgn w) * i16.u8 (v >> u8.i8 (i8.abs w))
    let maxWt = 2i8 -- TODO pass as a parameter
    in
    if w > 0 then
        if w == maxWt then 0 else i16.u8 (v >> u8.i8 w)
    else
        if w == -maxWt then 0 else - i16.u8 (v >> u8.i8 (-w))

def activation (s: i16): Val =
    -- ReLU
    if s > 0 then u8.i16 (i16.min s 255) else 0


-- input layer with j nodes -> output layer with k nodes
def outputs [k] [j] (inputs: [j]Val) (interWts: [k][j]Wt): [k]Val =
    interWts
    |> map (\inputWts ->
        reduce (+) 0 (map2 signedRightShift inputWts inputs)
    )
    -- "boosting" the sum by an arbitrary factor of 64 before dividing by the number of input nodes
     |> map (\s -> (s * 64) / i16.i64 j)
    |> map activation

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
    : ([n][i]Wt, [lmo][n][n]Wt, [o][n]Wt, i32, [o]Val) =
    foldl (\(iWts, hWts, oWts, goodAnswers, _) (x, y) ->
        let inputVals = outputs x iWts
        let hiddenVals = foldl outputs inputVals hWts
        let outputVals = outputs hiddenVals oWts
        let isGood =
            outputVals
            |> indexOfGreatest
            |> (==) (i64.u8 y)
        let teachCfg = { maxWt, learningStep, wasGood = isGood }
        in
        ( teachInter teachCfg iWts
        , hWts |> map (\wts -> teachInter teachCfg wts)
        , teachInter teachCfg oWts
        , goodAnswers + if isGood then 1 else 0
        , outputVals
        )
    )
    (inputWts, hiddenWts, outputWts, 0, (tabulate o (\_ -> 0u8)))
    (zip xs ys)

-- TODO
entry predict (x: i32): i32 =
    x + 42
