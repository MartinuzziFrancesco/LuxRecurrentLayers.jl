using LuxRecurrentLayers
using Documenter

DocMeta.setdocmeta!(LuxRecurrentLayers, :DocTestSetup, :(using LuxRecurrentLayers); recursive=true)

makedocs(;
    modules=[LuxRecurrentLayers],
    authors="Francesco Martinuzzi",
    sitename="LuxRecurrentLayers.jl",
    format=Documenter.HTML(;
        canonical="https://MartinuzziFrancesco.github.io/LuxRecurrentLayers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MartinuzziFrancesco/LuxRecurrentLayers.jl",
    devbranch="main",
)
