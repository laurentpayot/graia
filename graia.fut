-- Graia

type InputVal = u8
type HiddenVal = u8
type OutputVal = u8

type Weight = i8

-- i = inputs
-- n = neurons per layer
-- o = outputs
-- lmo = layers minus one
-- r = rows

type InputWs [n][i] = [n][i]Weight
type HiddenWs [lmo][n] = [lmo][n][n]Weight
type OutputWs [o][n] = [o][n]Weight

-- type Weights [i][n][lmo][o] = {
--   input: InputWs [n][i],
--   hidden: HiddenWs [lmo][n],
--   output: OutputWs [o][n] ,
-- }

entry fit [r][i][n][lmo][o]
  (inputWs: InputWs [n][i]) (hiddenWs: HiddenWs [lmo][n]) (outputWs: OutputWs [o][n])
  ( xs: [r][i]InputVal) (ys: [r]OutputVal) (epochs: i32)
  : (i64, i64, i64, i64, i64, i32) =
  -- https://futhark.readthedocs.io/en/latest/error-index.html#nonconstructive-entry
  -- TODO
  (r, i, n, lmo, o, epochs)

entry predict (x: i32): i32 =
  x + 42
