# This file contains the custom type for the Laplace operator with either
# Dirichlet or Neumann boundary conditions.

# FIXME: deprecated method of lu! being used
struct Laplace{Ny, Nz, LU}
    lus::Vector{LU}

    function Laplace(Ny::Int, Nz::Int, β::T, DD::AbstractMatrix{T}) where {T<:AbstractFloat}
        vec = [LinearAlgebra.lu!(_apply_BC!(DD - LinearAlgebra.I*(nz*β)^2), Val(false)) for nz in 0:(Nz >> 1)]
        new{Ny, Nz, eltype(vec)}(vec)
    end

    function Laplace(Ny::Int, Nz::Int, β::T, DD::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T<:AbstractFloat}
        vec = [LinearAlgebra.lu!(_apply_BC!(DD - LinearAlgebra.I*(nz*β)^2, D), Val(false)) for nz in 0:(Nz >> 1)]
        new{Ny, Nz, eltype(vec)}(vec)
    end
end

"""
    Modify the provided array to impose Dirichlet boundary conditions.
"""
function _apply_BC!(a::AbstractMatrix)
    a[1, :] = Eye(a[1, :], 1)
    a[end, :] = Eye(a[end, :])
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
    homogeneous boundary conditions imposed on the Laplace operator before
    passing as an argument.
"""
function solve!(phi::AbstractArray{T, 3}, laplace::Laplace{Ny, Nz}, rhs::AbstractArray{T, 3}) where {T, Ny, Nz}
    # extract temporal size
    _Nt = size(phi)[3]

    # initialise intermediate vectors
    _phi = Vector{T}(undef, Ny); _rhs = Vector{T}(undef, Ny)

    # loop over temporal and spanwise wavenumbers
    for nt in 1:_Nt, nz in 1:((Nz >> 1) + 1)
        # impose boundary conditions
        _rhs .= rhs[:, nz, nt]; _rhs[1] = _rhs[Ny] = 0

        # solve the poisson equation
        LinearAlgebra.ldiv!(_phi, laplace.lus[nz], _rhs)

        # assign the solution to the input matrix
        phi[:, nz, nt] .= _phi
    end

    return phi
end

"""
    Solve the Poisson equation for a 2D spatio-temporal scalar field with
    inhomogeneous boundary conditions imposed on the Laplace operator before
    passing as an argument.
"""
function solve!(phi::AbstractArray{T, 3}, laplace::Laplace{Ny, Nz}, rhs::AbstractArray{T, 3}, bc_data::NTuple{2, AbstractMatrix{T}}) where {T, Ny, Nz}
    # extract temporal size
    _Nt = size(phi)[3]

    # intialise intermediate vectors
    _phi = Vector{T}(undef, Ny); _rhs = Vector{T}(undef, Ny)

    # loop over temporal and spanwise wavenumbers
    for nt in 1:_Nt, nz in 1:((Nz >> 1) + 1)
        # impose boundary conditions
        _rhs .= rhs[:, nz, nt]
        _rhs[1] = bc_data[1][nz, nt]
        _rhs[Ny] = bc_data[2][nz, nt]

        # solve the poisson equation
        LinearAlgebra.ldiv!(_phi, laplace.lus[nz], _rhs)

        # assign the solution to the input matrix
        phi[:, nz, nt] .= _phi
    end

    return phi
end
