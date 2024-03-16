-- Graia

type InputVal = u8
type HiddenVal = u8
type OutputVal = u8

type Weight = i8

-- i = inputs
-- l = layers
-- n = neurons per layer
-- o = outputs
-- r = rows

type Weights [i][n][l][o] = {
  input: [n][i]Weight,
  hidden: [l-1][n][n]Weight,
  output: [o][n]Weight,
}

entry fit [r][i][n][l][o] (weights: Weights [i][n][l][o]) ( xs: [r][i]InputVal) (ys: [r]OutputVal) (epochs: i32): (i64, i64) =
  -- TODO
  (r, i)

entry predict (x: i32): i32 =
  x + 42
