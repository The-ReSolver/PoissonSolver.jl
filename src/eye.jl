# This file contains the definition to construct a lightweight object that acts
# like the standard cartesian basis vector of a given length.

struct Eye{L, N, T} <: AbstractVector{T}
    Eye(L::Int, N::Int, ::Type{T}=Float64) where {T} = new{L, N, T}()
end

Base.size(::Eye{L}) where {L} = (L,)
Base.IndexStyle(::Type{<:Eye}) = Base.IndexLinear()

function Base.getindex(::Eye{L, N, T}, i::Int) where {L, N, T}
    if i == N
        return T(1)
    else
        return T(0)
    end
end
