
def scanner [l] (a: ([l]i64, [l]i64)) (b: ([l]i64, [l]i64)) : ([l]i64, [l]i64) =
    (b.0, a.1)

-- def main (l: i64): [l]([l]i64, [l]i64) =
def main (l: i64): i64 =
  scan scanner
    ((tabulate l (\i -> i)), (tabulate l (\i -> i)))
    (tabulate l (\i  ->  ((tabulate l (\_ -> i)), (tabulate l (\_ -> i)))))
  |> (\layers -> layers[0])
  |> (\(a, _) -> a[0])
