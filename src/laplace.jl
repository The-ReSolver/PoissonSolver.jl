# This file contains the custom type for the Laplace operator with either
# Dirichlet or Neumann boundary conditions.

export Laplace

# TODO: How to enforce BCs?
# TODO: Unit tests for the construction of Chebyshev differentation matrices!
# TODO: Make it a custom subtype of AbstractVector and store as 3D array, use interface to define behaviour
# TODO: Add effect of z derivative to make it the full Laplace operator

struct Laplace{T, Nz, Ny}
    lus::Vector{Matrix{T}}

    function Laplace(Nz::Int, Ny::Int, ::Type{T}=Float64) where {T}
        # initialise an empty vector of matrices
        vec = Matrix{T}[]

        # loop over given dimensions
        for i in 1:Nz
            push!(vec, cheb_double_diffmat(Ny))
        end

        return new{T, Nz, Ny}(vec)
    end
end

"""
Calculate the second order Chebyshev differentiation matrix for a given number
of Chebyshev discretisation points.

U. Ehrenstein, R. Peyret (1989), A Chebyshev collocation method for the Navier-
Stokes equations with application to double-diffusive convection.
"""
function cheb_double_diffmat(Ny::Int)
    # initialise matrix
    diffmat = Matrix{Float64}(undef, Ny, Ny)

    # define anonymous function for c_k coefficient and x location
    c = i -> i == 1 || i == Ny ? 2 : 1
    x = i -> cos(((i - 1)*π)/(Ny - 1))

    # loop over matrix assigning values
    for i in 1:Ny, j in 1:Ny
        if i == j
            # evaluate diagonals
            diffmat[i, i] = -((((Ny - 1)^2 - 1)*(1 - x(i)^2) + 3)/(3*(1 - x(i)^2)^2))
        elseif i == 1
            # evaluate top edge (exclusive of diagonal corner)
            diffmat[1, j] = (2/3)*(((-1)^(j - 1))/c(j))*(((2*(Ny - 1)^2 + 1)*(1 - x(j)) - 6)/((1 - x(j))^2))
        elseif i == Ny
            # evaluate bottom edge (exclusive of diagonal corner)
            diffmat[Ny, j] = (2/3)*(((-1)^(j + Ny - 2))/c(j))*(((2*(Ny - 1)^2 + 1)*(1 + x(j)) - 6)/((1 + x(j))^2))
        else
            # evaluate everything else
            diffmat[i, j] = (((-1)^(i + j + 1))/c(j))*((2 - (x(i)*x(j)) - (x(i)^2))/((1 - (x(i)^2))*(x(i) - x(j))^2))
        end
    end
    diffmat[Ny, Ny] = diffmat[1, 1] = ((Ny - 1)^4 - 1)/15

    return diffmat
end

# function cheb_diffmat(Ny::Int, ::Type{T}=Float64) where {T<:Number}
#     diffmat = Matrix{T}(undef, Ny, Ny)
#     c = i -> i == 1 || i == Ny ? 2 : 1
#     x = i -> cos(((i - 1)*π)/(Ny - 1))
#     for i in 1:Ny, j in 1:Ny
#         if i == j
#             diffmat[i, i] = -x(i)/(2*(1 - x(j)^2))
#         else
#             diffmat[i, j] = (c(i)/c(j))*(((-1)^(i + j))/(x(i) - x(j)))
#         end
#     end
#     diffmat[1, 1] = (2*(Ny - 1)^2 + 1)/6
#     diffmat[Ny, Ny] = -diffmat[1, 1]

#     return diffmat
# end
