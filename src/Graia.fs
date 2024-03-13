module Graia

printfn "🌄 Graia v0.0.1"

open System.Runtime.Intrinsics
printfn "Vector128: %b" Vector128.IsHardwareAccelerated
printfn "Vector256: %b" Vector256.IsHardwareAccelerated
printfn "Vector512: %b" Vector512.IsHardwareAccelerated

open System.Collections

let a: BitArray = BitArray(3)
let b: BitArray = BitArray(3)

a.Set(1, true)
b.Set(2, true)
// let b = Seq.iter (fun x -> printfn $"%A{x}") a

// for x, y in Seq.zip a.GetEnumerator() b.GetEnumerator() do
//     printfn $"%A{x}"

for x in a do
    printfn $"%A{x}"

printfn $"%A{a}"
printfn $"%A{b}"
