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

    # -------------------------------------------------------------------------
    # Initialise Poisson solver
    # -------------------------------------------------------------------------
    # intiialise constants
    β2 = 1.0
    Ny2 = 50; Nz2 = 50; Nt2 = 50

    # initialise functions
    sol1_fun(y, z, t) = (1 - y^2)
    rhs1_fun(y, z, t) = -2.0

    # chebyshev points and differentiation matrix
    D2 = chebdiff(Ny2); DD2 = chebddiff(Ny2)
    y = chebpts(Ny2)

    # initialise grid
    grid = Grid(y, Nz2, Nt2, D2, DD2, rand(Ny2))

    # initialise Laplace operators
    laplace1 = Laplace(Ny2, Nz2, β2, DD2)
    laplace2 = Laplace(Ny2, Nz2, β2, DD2, D2)

    # -------------------------------------------------------------------------
    # Dirichlet
    # -------------------------------------------------------------------------
    # initialise functions
    sol5_fun(y, z, t) = (1 - y^2)*exp(cos(z))*atan(sin(t))
    rhs5_fun(y, z, t) = (-2*exp(cos(z)) + (sin(z)^2 - cos(z))*(1 - y^2)*exp(cos(z)))*atan(sin(t))

    # initialise solution field
    ϕ5_spec = SpectralField(grid)
    ϕ5_phys = PhysicalField(grid)

    # initialise FFT plans
    FFT = FFTPlan!(ϕ5_phys, flags = FFTW.ESTIMATE)
    IFFT = IFFTPlan!(ϕ5_spec, flags = FFTW.ESTIMATE)

    # initialise RHS field
    rhs5_spec = SpectralField(grid)
    rhs5_phys = PhysicalField(grid, rhs5_fun)

    # populate spectral version of RHS
    FFT(rhs5_spec, rhs5_phys)

    # find solution
    solve!(ϕ5_spec, laplace1, rhs5_spec)
    IFFT(ϕ5_phys, ϕ5_spec)

    @test ϕ5_phys ≈ PhysicalField(grid, sol5_fun)

    # -------------------------------------------------------------------------
    # Neumann
    # -------------------------------------------------------------------------
    # initialise functions
    sol6_fun(y, z, t) = y*(((y^2)/3) - 1)*exp(cos(z))*atan(sin(t))
    rhs6_fun(y, z, t) = (2*y*exp(cos(z)) + (sin(z)^2 - cos(z))*y*(((y^2)/3) - 1)*exp(cos(z)))*atan(sin(t))

    # initialise solution field
    ϕ6_spec = SpectralField(grid)
    ϕ6_phys = PhysicalField(grid)
    ϕ6_sol = PhysicalField(grid, sol6_fun)

    # initialise RHS field
    rhs6_spec = SpectralField(grid)
    rhs6_phys = PhysicalField(grid, rhs6_fun)

    # populate spectral version of RHS
    FFT(rhs6_spec, rhs6_phys)

    # find solution
    solve!(ϕ6_spec, laplace2, rhs6_spec)
    IFFT(ϕ6_phys, ϕ6_spec)

    # modify solution with offset (arbitrary when using Neumann BCs)
    for i in 1:Nt2
        ϕ6_phys[:, :, i] = ϕ6_phys[:, :, i] .- (ϕ6_phys[1, 1, i] - ϕ6_sol[1, 1, i])
    end

    @test ϕ6_phys ≈ PhysicalField(grid, sol6_fun)
end
