# This file contains the custom type for the Laplace operator with either
# Dirichlet or Neumann boundary conditions.

export Laplace, solve!

struct Laplace{Ny, Nz, LU}
    lus::Vector{LU}

    function Laplace(Ny::Int, Nz::Int, β::T, DD::AbstractMatrix{T}) where {T<:AbstractFloat}
        vec = [LinearAlgebra.lu!(_apply_BC!(DD - LinearAlgebra.I*(nz*β)^2), Val(false)) for nz in 0:(Nz >> 1)]
        new{Ny, Nz, eltype(vec)}(vec)
    end

    function Laplace(Ny::Int, Nz::Int, β::T, DD::AbstractMatrix{T}, D::AbstractMatrix{T}; noslip::Bool=false) where {T<:AbstractFloat}
        if noslip === false
            vec = [LinearAlgebra.lu!(_apply_BC!(DD - LinearAlgebra.I*(nz*β)^2, D), Val(false)) for nz in 0:(Nz >> 1)]
        else
            vec = [LinearAlgebra.lu!(_apply_noslip!(_apply_BC!(DD - LinearAlgebra.I*(nz*β)^2, D)), Val(false)) for nz in 0:(Nz >> 1)]
        end
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

function _apply_noslip!(a::AbstractMatrix)
    a[2, :] = Eye(a[2, :], 1)
    a[end - 1, :] = Eye(a[end - 1, :])
    Ny = size(a)[1]
    a = a[[2, 1, 3:(Ny - 2)..., Ny, Ny - 1], :]
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

function solve!(phi::AbstractArray{T, 3}, laplace::Laplace{Ny, Nz}, rhs::AbstractArray{T, 3}, ::Bool) where {T, Ny, Nz}
    # extract temporal size
    _Nt = size(phi)[3]

    # initialise intermediate vectors
    _phi = Vector{T}(undef, Ny); _rhs = Vector{T}(undef, Ny)

    # loop over temporal and spanwise wavenumbers
    for nt in 1:_Nt, nz in 1:((Nz >> 1) + 1)
        # impose boundary conditions
        _rhs .= rhs[:, nz, nt]; _rhs[1] = _rhs[end] = _rhs[2] = _rhs[end - 1] = 0

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
