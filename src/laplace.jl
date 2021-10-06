# This file contains the custom type for the Laplace operator with either
# Dirichlet or Neumann boundary conditions.

export Laplace

struct Laplace{T}
    lus<:Vector{T}

    # construct from size and type
    # Laplace(M::Int) =  new{M, Vector{T}}(zeros(M))
end
