open System.Runtime.InteropServices

module NN =
    [<DllImport("lib/nn.c", CallingConvention = CallingConvention.Cdecl)>]
    extern int32 main(int32)

let result = NN.main 1

printfn "Result: %d" result
