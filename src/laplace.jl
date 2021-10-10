# This file contains the custom type for the Laplace operator with either
# Dirichlet or Neumann boundary conditions.

export Laplace, solve!

struct Laplace{Ny, Nz, BC, LU}
    lus::Vector{LU}

    function Laplace(Nz::Int, Ny::Int, β::Float64, BC::Symbol, diffmat::AbstractMatrix=cheb_double_diffmat(Ny))
        # loop over spanwise wavenumbers and take LU decomposition of laplace operator
        vec = [LinearAlgebra.lu!(_apply_BC!(diffmat - LinearAlgebra.I*(nz*β)^2, BC), Val(false)) for nz in 0:Nz]

        return new{Ny, Nz, BC, eltype(vec)}(vec)
    end
end

"""
Modify the provided array to impose the given either Dirichlet or Neumann
boundary conditions.
"""
function _apply_BC!(a::AbstractMatrix, BC::Symbol)
    if BC == :Dirichlet
        # set first row to all zero except first element
        a[1, :] .= 0.0
        a[1, 1] = 1.0

        # set last row to all zero except first element
        a[end, :] .= 0.0
        a[end, end] = 1.0
    elseif BC == :Neumann
        # set the first and last row to the first row of the first order differentation matrix
        a[1, :] = cheb_single_diffmat(size(a, 1), 1)
        a[end, :] = cheb_single_diffmat(size(a, 1), size(a, 1))
    else
        throw(ArgumentError("Not a valid boundary condition: "*string(BC)))
    end

    return a
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
