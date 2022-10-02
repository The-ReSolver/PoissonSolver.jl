# This script performs a benchmarking of the Poisson solver.

using BenchmarkTools

using PoissonSolver
using Fields
# using ChebUtils
using FDGrids

# setup fields
β = 1.0
Ny = 64; Nz = 64; Nt = 64
y = gridpoints(Ny)
# D2 = chebdiff(Ny); DD2 = chebddiff(Ny)
D2 = DiffMatrix(y, 5, 1)
DD2 = DiffMatrix(y, 5, 2)
grid = Grid(y, Nz, Nt, D2, DD2, rand(Ny), β, 0.0)
laplace_banded = Laplace(Ny, Nz, β, DD2, D2, :banded)
laplace_lapack = Laplace(Ny, Nz, β, DD2, D2, :lapack)
ϕ = SpectralField(grid)
rhs = SpectralField(grid)
BC = (zeros(ComplexF64, Nz, Nt), zeros(ComplexF64, Nz, Nt))

@btime solve!($ϕ, $laplace_banded, $rhs, $BC)
@btime solve!($ϕ, $laplace_lapack, $rhs, $BC)

# NOTE: custom banded solver is faster for same grid size
# NOTE: diff matrix implementation is faster as plot would imply
