open System.Runtime.InteropServices

module NN =

    [<DllImport("lib/nn", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint futhark_context_config_new()

    [<DllImport("lib/nn", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint futhark_context_new(nativeint)

    [<DllImport("lib/nn", CallingConvention = CallingConvention.Cdecl)>]
    extern void futhark_context_sync(nativeint)

    [<DllImport("lib/nn", CallingConvention = CallingConvention.Cdecl)>]
    extern void futhark_new_i32_1d(nativeint, int, int)

    [<DllImport("lib/nn", CallingConvention = CallingConvention.Cdecl)>]
    extern int32 futhark_entry_foo(nativeint, nativeint, int32)

// [<DllImport("lib/nn", CallingConvention = CallingConvention.Cdecl)>]
// extern int32 futhark_entry_bar(int32)

open NN

printfn "futhark_new_i32_1d: %A" futhark_new_i32_1d

// printfn "foo: %A" futhark_entry_foo
// printfn "bar: %A" futhark_entry_bar


let cfg = futhark_context_config_new ()
printfn "cfg: %A" cfg
let ctx = futhark_context_new cfg
printfn "ctx: %A" ctx

// let foo (x: int32) : int32 =
//     let mutable res: nativeint = 0

//     let res2 = futhark_entry_foo (ctx, res, x)
//     futhark_context_sync (ctx)
//     res2

printfn "Starting…"

let mutable res: nativeint = 0
let res2 = futhark_entry_foo (ctx, 0, 100)

printfn "res: %A" res2


// printfn "foo result: %d" <| futhark_entry_foo 100
// printfn "bar result: %d" <| futhark_entry_bar 100

// printfn "foo result: %d" <| foo 100

printfn "Done!"
