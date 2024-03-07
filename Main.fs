open System.Runtime.Intrinsics.X86
open System.Runtime.Intrinsics
open System.Numerics

printfn "🌄 Graia v0.0.1"

printfn "Popcnt: %b" Popcnt.IsSupported
printfn "Vector: %b" Vector.IsHardwareAccelerated
printfn "Vector512: %b" Vector512.IsHardwareAccelerated
printfn "Vector256: %b" Vector256.IsHardwareAccelerated
printfn "Vector256<byte>.Count: %d" Vector256<byte>.Count
printfn "Vector256<uint64>.Count: %d" Vector256<uint64>.Count
