module Graia

printfn "🌄 Graia v0.0.1"

open System.Runtime.Intrinsics.X86
open System.Runtime.Intrinsics
open System.Numerics


printfn "Popcnt.X64: %b" Popcnt.X64.IsSupported
printfn "Vector: %b" Vector.IsHardwareAccelerated
printfn "Vector256: %b" Vector256.IsHardwareAccelerated
printfn "Vector512: %b" Vector512.IsHardwareAccelerated
