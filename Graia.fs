module Graia

open System.IO

open FSharpPlus
open System



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


type Row = string * byte seq


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

let toSvg (data: byte seq) : string =
    let array = Array.ofSeq data

    let mutable svg =
        """"<svg width="70" height="70" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">"""
        + "\n"

    for x = 0 to 27 do
        for y = 0 to 27 do
            let v = array[x * 28 + y]

            svg <-
                svg
                + $"""<rect x="{x}" y="{y}" width="1" height="1" fill="rgb({v},{v},{v})"/>"""

        svg <- svg + "\n"

    svg <- svg + "</svg>"
    svg
