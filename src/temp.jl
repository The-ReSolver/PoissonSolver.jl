# This file is to try and improve my ability to test my julia code without
# having to go in and out of the julia REPL.

include("laplace.jl")

a = Laplace(3, 3)
println(a)
# println(a.lus)
