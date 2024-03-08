module Graia

open System.IO

open FSharpPlus



printfn "🌄 Graia v0.0.1"

// open System.Runtime.Intrinsics.X86datasets/mnist_train.csv
// open System.Runtime.Intrinsics
// open System.Numerics

// printfn "Popcnt.X64: %b" Popcnt.X64.IsSupported
// printfn "Vector: %b" Vector.IsHardwareAccelerated
// printfn "Vector512: %b" Vector512.IsHardwareAccelerated
// printfn "Vector256: %b" Vector256.IsHardwareAccelerated
// printfn "Vector256<byte>.Count: %d" Vector256<byte>.Count
// printfn "Vector256<uint64>.Count: %d" Vector256<uint64>.Count

type RowLabel = string
type RowData = byte seq
type Row = RowLabel * RowData


let loadMnist (path: string) : Row array =
    File.ReadAllText(path)
    |> String.split [ "\n" ]
    // remove header row
    |> Seq.skip 1
    |> Seq.map (String.split [ "," ])
    |> fold
        (fun acc row ->
            let label = Seq.head row
            let data = Seq.skip 1 row |> Seq.map byte
            Array.append acc [| (label, data) |])
        [||]

// print a 28x28 svg representation of the 784x1 byte array
let toSvg (data: byte seq) = data |> Seq.chunkBySize 28
// TODO
// |> sprintf
//     "<svg width=\"28\" height=\"28\" viewBox=\"0 0 784 28\" xmlns=\"http://www.w3.org/2000/svg\">\n%s\n</svg>"
