module PoissonSolver

using LinearAlgebra

include("laplace.jl")

greet() = println("Hello World!")
greet_number() = println("Hello ", tr([1 2; 3 4]))

end
