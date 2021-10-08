# This file contains the custom type for the Laplace operator with either
# Dirichlet or Neumann boundary conditions.

import LinearAlgebra

export Laplace, solve!

# TODO: Unit tests for the construction of Chebyshev differentation matrices!

struct Laplace{Ny, Nz, BC, LU}
    lus::Vector{LU}

    function Laplace(Nz::Int, Ny::Int, BC::Symbol, diffmat::AbstractMatrix=cheb_double_diffmat(Ny))
        # initialise an empty vector of matrices
        vec = [LinearAlgebra.lu!(apply_BC!(diffmat - I*(nz*β)^2, BC)) for nz in 0:Nz]

        return new{Ny, Nz, BC, eltype(vec)}(vec)
    end
end

"""
Modify the provided array to impose the given either Dirichlet or Neumann
boundary conditions.
"""
function apply_BC!(a::AbstractMatrix, BC::Symbol)
    if BC == :Dirichlet
        # set first row to all zero except first element
        a[1, :] .= 0
        a[1, 1] = 1

        # set last row to all zero except first element
        a[end, :] .= 0
        a[end, end] = 1
    elseif BC == :Neumann
        # set the first and last row to the first row of the first order differentation matrix
        a[1, :] = cheb_diffmat(size(a, 1), 1)
        a[end, :] = cheb_diffmat(size(a, 1), size(a, 1))
    else
        throw(ArgumentError("Not a valid boundary condition: "*string(BC)))
    end
end

"""
Solve the Poisson equation for a 2D spatio-temporal scalar field with boundary
conditions imposed on the Laplace operator before passing as an argument. Only
homogeneous Dirichlet or Neumann boundary conditions are treated.
"""
function solve!(phi::AbstractArray{T, 3}, laplace::Laplace{Ny}, rhs::AbstractArray{T, 3}) where {T, Ny}
    # initialise intermediate vectors to minimise memory assignment
    _phi = Vector{T}(undef, Ny); _rhs = Vector{T}(undef, Ny)

    # loop over temporal and spanwise wavenumbers
    # THE ZERO-ZERO MODE IS DISCARDED!
    for nt in 0:Nt, nz in 0:Nz
        # impose boundary conditions on RHS of equation
        _rhs .= rhs[:, nz, nt]; _rhs[1] = rhs[Ny] = 0

        # solve the poisson equation
        LinearAlgebra.ldiv!(_phi, laplace.lus[nz], _rhs)

        # assign the solution to the input matrix
        phi[:, nz, nt] .= _phi
    end
end

"""
Calculate the second order Chebyshev differentiation matrix for a given number
of Chebyshev discretisation points.

U. Ehrenstein, R. Peyret (1989), A Chebyshev collocation method for the Navier-
Stokes equations with application to double-diffusive convection.
"""
function cheb_double_diffmat(Ny::Int, ::Type{T}=Float64) where {T}
    # initialise matrix
    diffmat = Matrix{T}(undef, Ny, Ny)

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

    # re-evaluate the diagonal corners
    diffmat[Ny, Ny] = diffmat[1, 1] = ((Ny - 1)^4 - 1)/15

    return diffmat
end

"""
Calculate the first order Chebyshev differentiation matrix for a given number
of Chebyshev discretisation points.

U. Ehrenstein, R. Peyret (1989), A Chebyshev collocation method for the Navier-
Stokes equations with application to double-diffusive convection.
"""
function cheb_diffmat(Ny::Int, ::Type{T}=Float64) where {T}
    # initialise matrix
    diffmat = Matrix{T}(undef, Ny, Ny)

    # define anonymous function for c_k coefficient and x location
    c = i -> i == 1 || i == Ny ? 2 : 1
    x = i -> cos(((i - 1)*π)/(Ny - 1))

    # loop over matrix assigning values
    for i in 1:Ny, j in 1:Ny
        if i == j
            # evaluate diagonals
            diffmat[i, i] = -x(i)/(2*(1 - x(j)^2))
        else
            # evaluate everything else
            diffmat[i, j] = (c(i)/c(j))*(((-1)^(i + j))/(x(i) - x(j)))
        end
    end

    # re-evaluate the diagonal corners
    diffmat[1, 1] = (2*(Ny - 1)^2 + 1)/6
    diffmat[Ny, Ny] = -diffmat[1, 1]

    return diffmat
end

# TODO: Re-implement swapping the hierarchy to make more efficient
"""
Calculate the first row of a first order Chebyshev differentiation matrix for a
given number of Chebyshev discretisation points.

U. Ehrenstein, R. Peyret (1989), A Chebyshev collocation method for the Navier-
Stokes equations with application to double-diffusive convection.
"""
function cheb_diffmat(Ny::Int, row::Int, ::Type{T}=Float64) where {T}
    return cheb_diffmat(Ny::Int, T)[row, :]
end
