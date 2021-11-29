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
    Ny2 = 64; Nz2 = 64; Nt2 = 64

    # chebyshev points and differentiation matrix
    D2 = chebdiff(Ny2); DD2 = chebddiff(Ny2)
    y = chebpts(Ny2)

    # initialise grid
    grid = Grid(y, Nz2, Nt2, D2, DD2, rand(Ny2))

    # initialise Laplace operators
    laplace1 = Laplace(Ny2, Nz2, β2, DD2)
    laplace2 = Laplace(Ny2, Nz2, β2, DD2, D2)

    # -------------------------------------------------------------------------
    # Dirichlet (homogeneous BC)
    # -------------------------------------------------------------------------
    # initialise functions
    sol1_fun(y, z, t) = (1 - y^2)*exp(cos(z))*atan(sin(t))
    rhs1_fun(y, z, t) = (-2*exp(cos(z)) + (sin(z)^2 - cos(z))*(1 - y^2)*exp(cos(z)))*atan(sin(t))

    # # initialise solution field
    ϕ1_spec = SpectralField(grid)
    ϕ1_phys = PhysicalField(grid)

    # # initialise FFT plans
    FFT = FFTPlan!(ϕ1_phys, flags = FFTW.ESTIMATE)
    IFFT = IFFTPlan!(ϕ1_spec, flags = FFTW.ESTIMATE)

    # # initialise RHS field
    rhs1_spec = SpectralField(grid)
    rhs1_phys = PhysicalField(grid, rhs1_fun)

    # populate spectral version of RHS
    FFT(rhs1_spec, rhs1_phys)

    # find solution
    solve!(ϕ1_spec, laplace1, rhs1_spec)
    IFFT(ϕ1_phys, ϕ1_spec)

    @test ϕ1_phys ≈ PhysicalField(grid, sol1_fun)

    # -------------------------------------------------------------------------
    # Neumann (homogeneous BC)
    # -------------------------------------------------------------------------
    # initialise functions
    sol2_fun(y, z, t) = y*(((y^2)/3) - 1)*exp(cos(z))*atan(sin(t))
    rhs2_fun(y, z, t) = (2*y*exp(cos(z)) + (sin(z)^2 - cos(z))*y*(((y^2)/3) - 1)*exp(cos(z)))*atan(sin(t))

    # initialise solution field
    ϕ2_spec = SpectralField(grid)
    ϕ2_phys = PhysicalField(grid)
    ϕ2_sol = PhysicalField(grid, sol2_fun)

    # initialise RHS field
    rhs2_spec = SpectralField(grid)
    rhs2_phys = PhysicalField(grid, rhs2_fun)

    # populate spectral version of RHS
    FFT(rhs2_spec, rhs2_phys)

    # find solution
    solve!(ϕ2_spec, laplace2, rhs2_spec)
    IFFT(ϕ2_phys, ϕ2_spec)

    # modify solution with offset (arbitrary when using Neumann BCs)
    for i in 1:Nt2
        ϕ2_phys[:, :, i] = ϕ2_phys[:, :, i] .- (ϕ2_phys[1, 1, i] - ϕ2_sol[1, 1, i])
    end

    @test ϕ2_phys ≈ PhysicalField(grid, sol2_fun)

    # -------------------------------------------------------------------------
    # Dirichlet (inhomogeneous BC)
    # -------------------------------------------------------------------------
    # initialise functions
    sol3_fun(y, z, t) = (2.0 - y^2)*exp(cos(z))*atan(sin(t))
    rhs3_fun(y, z, t) = (-2.0*exp(cos(z)) + (sin(z)^2 - cos(z))*(2 - y^2)*exp(cos(z)))*atan(sin(t))
    BC_dir_fun(y, z, t) = exp(cos(z))*atan(sin(t))

    # initialise solution field
    ϕ3_spec = SpectralField(grid)
    ϕ3_phys = PhysicalField(grid)

    # initialise RHS field
    rhs3_spec = SpectralField(grid)
    rhs3_phys = PhysicalField(grid, rhs3_fun)

    # populate spectral version of RHS
    FFT(rhs3_spec, rhs3_phys)

    # intialise boundary condition
    BC_dir_phys = PhysicalField(grid, BC_dir_fun)
    BC_dir_spec = SpectralField(grid)
    FFT(BC_dir_spec, BC_dir_phys)
    BC_dir = (BC_dir_spec[1, :, :], BC_dir_spec[1, :, :])

    # find solution
    solve!(ϕ3_spec, laplace1, rhs3_spec, BC_dir)
    IFFT(ϕ3_phys, ϕ3_spec)

    @test ϕ3_phys ≈ PhysicalField(grid, sol3_fun)

    # -------------------------------------------------------------------------
    # Neumann (inhomogeneous BC)
    # -------------------------------------------------------------------------
    # initialise functions
    sol4_fun(y, z, t) = y*(y^2 - 2)*exp(cos(z))*atan(sin(t))
    rhs4_fun(y, z, t) = (6*y*exp(cos(z)) + (sin(z)^2 - cos(z))*y*(y^2 - 2)*exp(cos(z)))*atan(sin(t))
    # NOTE: this has to be negative to work properly
    BC_neu_fun(y, z, t) = -exp(cos(z))*atan(sin(t))

    # initialise solution field
    ϕ4_spec = SpectralField(grid)
    ϕ4_phys = PhysicalField(grid)
    ϕ4_sol = PhysicalField(grid, sol4_fun)

    # initialise RHS field
    rhs4_spec = SpectralField(grid)
    rhs4_phys = PhysicalField(grid, rhs4_fun)

    # populate spectral version of RHS
    FFT(rhs4_spec, rhs4_phys)

    # initialise boundary condition
    BC_neu_phys = PhysicalField(grid, BC_neu_fun)
    BC_neu_spec = SpectralField(grid)
    FFT(BC_neu_spec, BC_neu_phys)
    BC_neu = (BC_neu_spec[1, :, :], BC_neu_spec[1, :, :])

    # find solution
    solve!(ϕ4_spec, laplace2, rhs4_spec, BC_neu)
    IFFT(ϕ4_phys, ϕ4_spec)

    # modify solution with offset (arbitrary when using Neumann BCs)
    for i in 1:Nt2
        ϕ4_phys[:, :, i] = ϕ4_phys[:, :, i] .- (ϕ4_phys[1, 1, i] - ϕ4_sol[1, 1, i])
    end

    @test ϕ4_phys ≈ PhysicalField(grid, sol4_fun)
end
