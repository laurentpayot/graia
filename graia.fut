-- Graia

type InputVal = u8
type HiddenVal = u8
type OutputVal = u8

-- W = weight
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

def feedForward [i][n][lmo][o]
  (model: Model [i][n][lmo][o]) (x: [i]InputVal) : (Model [i][n][lmo][o], OutputVal) =
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
  (model.inputWs, model.hiddenWs, model.outputWs, 0.0)

entry predict (x: i32): i32 =
  x + 42
