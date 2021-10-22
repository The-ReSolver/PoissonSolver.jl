# This file contains the test for the custom (discrete) Laplace operator type.

@testset "Laplace operator tests" begin
    # generate random numbers
    Ny = rand(2:50)
    Nz = rand(2:50)
    β = abs(randn())
    D = chebdiff(Ny)
    DD = chebddiff(Ny)
    rinner = rand(2:(Ny - 1))

    # initialise operators
    laplace_dirichlet = Laplace(Ny, Nz, β, DD)
    laplace_neumann = Laplace(Ny, Nz, β, DD, D)

    # correct size
    @test size(laplace_dirichlet.lus) == ((Nz >> 1) + 2, )
    @test size(laplace_neumann.lus) == ((Nz >> 1) + 2, )
    @test size(laplace_dirichlet.lus[1]) == (Ny, Ny)
    @test size(laplace_neumann.lus[1]) == (Ny, Ny)

    # boundary conditions applied correctly
    eye1 = zeros(Ny); eye2 = copy(eye1)
    eye1[1] = 1.0; eye2[end] = 1.0
    laplace_dirichlet_recon = laplace_dirichlet.lus[1].L*laplace_dirichlet.lus[1].U
    laplace_neumann_recon = laplace_neumann.lus[1].L*laplace_neumann.lus[1].U
    @test laplace_dirichlet_recon[1, :] == eye1
    @test laplace_dirichlet_recon[end, :] == eye2
    @test laplace_neumann_recon[1, :] ≈ D[1, :]
    @test laplace_neumann_recon[end, :] ≈ D[end, :]

    # are the inner (non-boundary condition values correct)
    double_diffmat = chebddiff(Ny)
    @test laplace_dirichlet_recon[rinner, :] ≈ double_diffmat[rinner, :]
    @test laplace_neumann_recon[rinner, :] ≈ double_diffmat[rinner, :]

    # can a correct solution be found
    # To test this I need to:
    #   - e^cos(x) as complicated function?


    # -------------------------------------------------------------------------
    # Simple Dirichlet
    # -------------------------------------------------------------------------
    # initialise constants and solution function
    sol1_fun(y, z, t) = (1 - y^2)
    rhs1_fun(y, z, t) = -2.0
    β2 = abs(randn())
    Ny2 = 64; Nz2 = 2; Nt2 = 2

    # chebyshev points and differentiation matrix
    D2 = chebdiff(Ny2); DD2 = chebddiff(Ny2)
    y = chebpts(Ny2)

    # initialise grid
    grid = Grid(y, Nz2, Nt2, D2, DD2, rand(Ny2))

    # initialise solution field
    ϕ1_spec = SpectralField(grid)
    ϕ1_phys = PhysicalField(grid)

    # initialise RHS field
    rhs1_spec = SpectralField(grid)
    rhs1_phys = PhysicalField(grid, rhs1_fun)

    # initialise FFT plans
    FFT = FFTPlan!(ϕ1_phys, flags = FFTW.ESTIMATE)
    IFFT = IFFTPlan!(ϕ1_spec, flags = FFTW.ESTIMATE)

    # populate spectral version of RHS
    FFT(rhs1_spec, rhs1_phys)

    # initialise Laplace operator
    laplace1 = Laplace(Ny2, Nz2, β2, DD2)

    # find solution
    solve!(ϕ1_spec, laplace1, rhs1_spec)
    IFFT(ϕ1_phys, ϕ1_spec)
    @test ϕ1_phys ≈ PhysicalField(grid, sol1_fun)

    # -------------------------------------------------------------------------
    # Simple Neumann
    # -------------------------------------------------------------------------
    # initialise constants and solution function
    sol2_fun(y, z, t) = y*(((y^2)/3) - 1)
    rhs2_fun(y, z, t) = 2*y

    # initialise solution field
    ϕ2_spec = SpectralField(grid)
    ϕ2_phys = PhysicalField(grid)

    # initialise RHS field
    rhs2_spec = SpectralField(grid)
    rhs2_phys = PhysicalField(grid, rhs2_fun)

    # populate spectral version of RHS
    FFT(rhs2_spec, rhs2_phys)

    # initialise Laplace operator
    laplace2 = Laplace(Ny2, Nz2, β2, DD2, D2)

    # find solution
    solve!(ϕ2_spec, laplace2, rhs2_spec)
    IFFT(ϕ2_phys, ϕ2_spec)

    # modify solution with offset (arbitrary when using Neumann BCs)
    ϕ2_phys = ϕ2_phys .- (ϕ2_phys[1, 1, 1] - PhysicalField(grid, sol2_fun)[1, 1, 1])
    @test ϕ2_phys ≈ PhysicalField(grid, sol2_fun)
end