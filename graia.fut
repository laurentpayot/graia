-- Graia

-- V = Value (input, hidden and output)
type V = u8
-- W = Weight
type W = i8

-- i = inputs
-- n = neurons per layer
-- o = outputs
-- lmo = layers minus one
-- r = rows

type Model [i][n][lmo][o] = {
  inputWs: [n][i]W,
  hiddenWs: [lmo][n][n]W,
  outputWs: [o][n]W
}

def signedRightShift (w: W) (v: V): i8 =
  if w == 0 then 0 else
    if w > 0 then v >> u8.i8 w else - (v >> (u8.i8 (-w)))

def getOutput [j] (ws: [j]W) (is: [j]V) : V =
  (zip ws is)
  |> map (\(w,i) -> w*i)
  foldl (+) 0

def feedForward [i][n][lmo][o]
  (model: Model [i][n][lmo][o]) (wasGood: bool) (inputs: [i]InputVal)
  : (Model [i][n][lmo][o], [o]OutputVal) =
  -- TODO
  (model, 0)

entry fit [r][i][n][lmo][o]
  (inputWs: [n][i]W) (hiddenWs: [lmo][n][n]W) (outputWs: [o][n]W)
  ( xs: [r][i]InputVal) (ys: [r]OutputVal)
  : ([n][i]W, [lmo][n][n]W, [o][n]W, f16) =
  -- TODO
  let model: Model [i][n][lmo][o] = {
    inputWs = inputWs,
    hiddenWs = hiddenWs,
    outputWs =outputWs
  } in
  zip xs ys
  |> map (\x y -> feedForward model true x)


  -- (model.inputWs, model.hiddenWs, model.outputWs, 0.0)

entry predict (x: i32): i32 =
  x + 42
