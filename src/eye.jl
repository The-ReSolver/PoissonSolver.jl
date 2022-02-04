# This file contains the definition to construct a lightweight object that acts
# like the standard cartesian basis vector of a given length.

struct Eye{L, N, T} <: AbstractVector{T}
    Eye(L::Int, N::Int, ::Type{T}=Float64) where {T} = (N > L) ? throw(ArgumentError("bruh")) : new{L, N, T}()
end

Eye(A::AbstractVector{T}, N) where {T} = Eye(length(A), N, T)
Eye(A::AbstractVector{T}) where {T} = Eye(A, length(A))

Base.size(::Eye{L}) where {L} = (L,)
Base.IndexStyle(::Type{<:Eye}) = Base.IndexLinear()
Base.getindex(e::Eye{L, N, T}, i::Int) where {L, N, T} = (0 < i < L + 1 || throw(BoundsError(e, i)); i == N ? T(1) : T(0))
