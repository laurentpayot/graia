-- Graia

-- TODO add model parameter
entry fit [n] (xs: [n][]u8, ys: [n]u8, epochs: i32): (i64, i64) =
  -- TODO
  (n, n)

entry predict (x: i32): i32 =
  x + 42

-- def main (x: []i32) (y: []i32): i32 =
--   reduce (+) 0 (map2 (*) x y)
