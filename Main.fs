open System.Runtime.InteropServices

module NN =
    [<DllImport("lib/nn")>]
    extern int32 main(int32)

let result = NN.main 1

printfn "Result: %d" result
