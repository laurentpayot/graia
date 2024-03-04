open System.Runtime.InteropServices

module NN =
    [<DllImport("lib/nn", CallingConvention = CallingConvention.Cdecl)>]
    extern int32 futhark_entry_foo(int32)

    extern int32 futhark_entry_bar()


printfn "foo: %A" NN.futhark_entry_foo
printfn "bar: %A" NN.futhark_entry_bar


printfn "foo result: %d" <| NN.futhark_entry_foo 100
printfn "bar result: %d" <| NN.futhark_entry_bar ()

printfn "Done"
