using IRBP
using Documenter

DocMeta.setdocmeta!(IRBP, :DocTestSetup, :(using IRBP); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [IRBP],
    authors = "Nathan Allaire <nathan.allaire@polymtl.ca>",
    repo = "https://github.com/nathanemac/IRBP.jl/blob/{commit}{path}#{line}",
    sitename = "IRBP.jl",
    format = Documenter.HTML(; canonical = "https://nathanemac.github.io/IRBP.jl"),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/nathanemac/IRBP.jl")
