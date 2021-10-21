# This file contains the custom type for the Laplace operator with either
# Dirichlet or Neumann boundary conditions.

export Laplace, solve!

struct Laplace{Ny, Nz, LU}
    lus::Vector{LU}

    function Laplace(Nz::Int, Ny::Int, β::T, DD::AbstractMatrix{T}) where {T<:AbstractFloat}
        vec = [LinearAlgebra.lu!(_apply_BC!(DD - LinearAlgebra.I*(nz*β)^2), Val(false)) for nz in 0:((Nz >> 1) + 1)]
        new{Ny, Nz, eltype(vec)}(vec)
    end

    function Laplace(Nz::Int, Ny::Int, β::T, DD::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T<:AbstractFloat}
        vec = [LinearAlgebra.lu!(_apply_BC!(DD - LinearAlgebra.I*(nz*β)^2, D), Val(false)) for nz in 0:((Nz >> 1) + 1)]
        new{Ny, Nz, eltype(vec)}(vec)
    end
end

"""
    Modify the provided array to impose Dirichlet boundary conditions.
"""
function _apply_BC!(a::AbstractMatrix)
    a[1, :] .= 0.0
    a[1, 1] = 1.0
    a[end, :] .= 0.0
    a[end, end] = 1.0
    return a
end

"""
    Modify the provided array to impose Neumann boundary conditions.
"""
function _apply_BC!(a::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
    a[1, :] = D[1, :]
    a[end, :] = D[end, :]
    return a
end

"""
    Solve the Poisson equation for a 2D spatio-temporal scalar field with
    boundary conditions imposed on the Laplace operator before passing as an
    argument. Only homogeneous Dirichlet or Neumann boundary conditions are
    treated.
"""
function solve!(phi::AbstractArray{T, 3}, laplace::Laplace{Ny, Nz, Nt}, rhs::AbstractArray{T, 3}) where {T, Ny, Nz, Nt}
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

    return phi
end
