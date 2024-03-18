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


def signedRightShift (w: Wt) (v: Val): i8 =
  -- (i8.sgn w) * i8.u8 (v >> u8.i8 (i8.abs w))
  i8.u8 <|
    if w == 0 then
      0
    else
      if w > 0 then
        v >> u8.i8  w
      else
        -(v >> u8.i8 (-w))

-- def getOutput [j] (ws: [j]Wt) (is: [j]Val) : Val =
--   (zip ws is)
--   |> map (\(w,i) -> w*i)
--   foldl (+) 0

-- def feedForward [i][n][lmo][o]
--   (model: Model [i][n][lmo][o]) (wasGood: bool) (inputs: [i]Val)
--   : (Model [i][n][lmo][o], [o]Val) =
--   -- TODO
--   (model, 0)

entry fit [r][i][n][lmo][o]
  (inputWts: [n][i]Wt) (hiddenWts: [lmo][n][n]Wt) (outputWts: [o][n]Wt)
  ( xs: [r][i]Val) (ys: [r]Val) (learningStep: u8)
  : ([n][i]Wt, [lmo][n][n]Wt, [o][n]Wt, f16) =
  -- TODO
  let model: Model [i][n][lmo][o] = {
    inputWts = inputWts,
    hiddenWts = hiddenWts,
    outputWts =outputWts
  } in
  -- zip xs ys |> map (\x y -> feedForward model true x)
  (model.inputWts, model.hiddenWts, model.outputWts, 0.0)

entry predict (x: i32): i32 =
  x + 42
