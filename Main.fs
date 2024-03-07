printfn "🌄 Graia v0.0.1"


open System.Runtime.Intrinsics.X86

if not (Popcnt.IsSupported) then
    printfn "Popcnt is not supported"
    exit 1

let n = 7UL


let c = Popcnt.X64.PopCount n


printfn "%d" c
