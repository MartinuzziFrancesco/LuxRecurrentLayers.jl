using Documenter, LuxRecurrentLayers
include("pages.jl")

mathengine = Documenter.MathJax()
DocMeta.setdocmeta!(LuxRecurrentLayers, :DocTestSetup, :(using LuxRecurrentLayers); recursive=true)

makedocs(;
    modules=[LuxRecurrentLayers],
    authors="Francesco Martinuzzi",
    sitename="LuxRecurrentLayers.jl",
    linkcheck=true,
    clean=true,
    format=Documenter.HTML(;
        canonical="https://MartinuzziFrancesco.github.io/LuxRecurrentLayers.jl",
        edit_link="main",
        assets=["assets/favicon.ico"],
    ),
    pages=pages,
)

deploydocs(;
    repo="github.com/MartinuzziFrancesco/LuxRecurrentLayers.jl",
    devbranch="main",
    push_preview=true
)
