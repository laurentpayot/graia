open System.Runtime.InteropServices

module NN =
    [<DllImport("lib/nn", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)>]
    extern int32 main(int32)

let result = NN.main 1

printfn "Result: %d" result
