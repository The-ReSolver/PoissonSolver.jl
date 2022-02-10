@testset "Laplace operator initialisation       " begin
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
    @test size(laplace_dirichlet.lus) == ((Nz >> 1) + 1,)
    @test size(laplace_neumann.lus) == ((Nz >> 1) + 1,)
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
end

@testset "Dirichlet homogeneous BC solution     " begin
    # initialise constants
    β = 1.0
    Ny = 64; Nz = 64; Nt = 64

    # chebyshev points and differentiation matrix
    D2 = chebdiff(Ny); DD2 = chebddiff(Ny)
    y = chebpts(Ny)

    # initialise grid
    grid = Grid(y, Nz, Nt, D2, DD2, rand(Ny), β, 0.0)

    # initialise Laplace operators
    laplace = Laplace(Ny, Nz, β, DD2)

    # initialise FFT plans
    FFT = FFTPlan!(grid, flags = FFTW.ESTIMATE)
    IFFT = IFFTPlan!(grid, flags = FFTW.ESTIMATE)

    # initialise functions
    sol_fun(y, z, t) = (1 - y^2)*exp(cos(z))*atan(sin(t))
    rhs_fun(y, z, t) = (-2*exp(cos(z)) + (sin(z)^2 - cos(z))*(1 - y^2)*exp(cos(z)))*atan(sin(t))

    # initialise solution field
    ϕ_spec = SpectralField(grid)
    ϕ_phys = PhysicalField(grid)

    # initialise RHS field
    rhs_spec = SpectralField(grid)
    rhs_phys = PhysicalField(grid, rhs_fun)

    # populate spectral version of RHS
    FFT(rhs_spec, rhs_phys)

    # find solution
    solve!(ϕ_spec, laplace, rhs_spec)
    IFFT(ϕ_phys, ϕ_spec)

    @test ϕ_phys ≈ PhysicalField(grid, sol_fun)
end

@testset "Neumann homogeneous BC solution       " begin
    # initialise constants
    β = 1.0
    Ny = 64; Nz = 64; Nt = 64

    # chebyshev points and differentiation matrix
    D2 = chebdiff(Ny); DD2 = chebddiff(Ny)
    y = chebpts(Ny)

    # initialise grid
    grid = Grid(y, Nz, Nt, D2, DD2, rand(Ny), β, 0.0)

    # initialise Laplace operators
    laplace = Laplace(Ny, Nz, β, DD2, D2)

    # initialise FFT plans
    FFT = FFTPlan!(grid, flags = FFTW.ESTIMATE)
    IFFT = IFFTPlan!(grid, flags = FFTW.ESTIMATE)

    # initialise functions
    sol_fun(y, z, t) = y*(((y^2)/3) - 1)*exp(cos(z))*atan(sin(t))
    rhs_fun(y, z, t) = (2*y*exp(cos(z)) + (sin(z)^2 - cos(z))*y*(((y^2)/3) - 1)*exp(cos(z)))*atan(sin(t))

    # initialise solution field
    ϕ_spec = SpectralField(grid)
    ϕ_phys = PhysicalField(grid)
    ϕ_sol = PhysicalField(grid, sol_fun)

    # initialise RHS field
    rhs_spec = SpectralField(grid)
    rhs_phys = PhysicalField(grid, rhs_fun)

    # populate spectral version of RHS
    FFT(rhs_spec, rhs_phys)

    # find solution
    solve!(ϕ_spec, laplace, rhs_spec)
    IFFT(ϕ_phys, ϕ_spec)

    # modify solution with offset (arbitrary when using Neumann BCs)
    for i in 1:Nt
        ϕ_phys[:, :, i] = ϕ_phys[:, :, i] .- (ϕ_phys[1, 1, i] - ϕ_sol[1, 1, i])
    end

    @test ϕ_phys ≈ ϕ_sol
end

@testset "Dirichlet inhomogeneous BC solution   " begin
    # initialise constants
    β = 1.0
    Ny = 64; Nz = 64; Nt = 64

    # chebyshev points and differentiation matrix
    D2 = chebdiff(Ny); DD2 = chebddiff(Ny)
    y = chebpts(Ny)

    # initialise grid
    grid = Grid(y, Nz, Nt, D2, DD2, rand(Ny), β, 0.0)

    # initialise Laplace operators
    laplace = Laplace(Ny, Nz, β, DD2)

    # initialise FFT plans
    FFT = FFTPlan!(grid, flags = FFTW.ESTIMATE)
    IFFT = IFFTPlan!(grid, flags = FFTW.ESTIMATE)

    # initialise functions
    sol_fun(y, z, t) = (2.0 - y^2)*exp(cos(z))*atan(sin(t))
    rhs_fun(y, z, t) = (-2.0*exp(cos(z)) + (sin(z)^2 - cos(z))*(2 - y^2)*exp(cos(z)))*atan(sin(t))
    BC_dir_fun(y, z, t) = exp(cos(z))*atan(sin(t))

    # initialise solution field
    ϕ_spec = SpectralField(grid)
    ϕ_phys = PhysicalField(grid)

    # initialise RHS field
    rhs_spec = SpectralField(grid)
    rhs_phys = PhysicalField(grid, rhs_fun)

    # populate spectral version of RHS
    FFT(rhs_spec, rhs_phys)

    # intialise boundary condition
    BC_dir_phys = PhysicalField(grid, BC_dir_fun)
    BC_dir_spec = SpectralField(grid)
    FFT(BC_dir_spec, BC_dir_phys)
    BC_dir = (BC_dir_spec[1, :, :], BC_dir_spec[1, :, :])

    # find solution
    solve!(ϕ_spec, laplace, rhs_spec, BC_dir)
    IFFT(ϕ_phys, ϕ_spec)

    @test ϕ_phys ≈ PhysicalField(grid, sol_fun)
end

@testset "Neumann inhomogeneous BC solution     " begin
    # initialise constants
    β = 1.0
    Ny = 64; Nz = 64; Nt = 64

    # chebyshev points and differentiation matrix
    D2 = chebdiff(Ny); DD2 = chebddiff(Ny)
    y = chebpts(Ny)

    # initialise grid
    grid = Grid(y, Nz, Nt, D2, DD2, rand(Ny), β, 0.0)

    # initialise Laplace operators
    laplace = Laplace(Ny, Nz, β, DD2, D2)

    # initialise FFT plans
    FFT = FFTPlan!(grid, flags = FFTW.ESTIMATE)
    IFFT = IFFTPlan!(grid, flags = FFTW.ESTIMATE)

    # initialise functions
    sol_fun(y, z, t) = y*(y^2 - 2)*exp(cos(z))*atan(sin(t))
    rhs_fun(y, z, t) = (6*y*exp(cos(z)) + (sin(z)^2 - cos(z))*y*(y^2 - 2)*exp(cos(z)))*atan(sin(t))
    BC_neu_fun(y, z, t) = exp(cos(z))*atan(sin(t))

    # initialise solution field
    ϕ_spec = SpectralField(grid)
    ϕ_phys = PhysicalField(grid)
    ϕ_sol = PhysicalField(grid, sol_fun)

    # initialise RHS field
    rhs_spec = SpectralField(grid)
    rhs_phys = PhysicalField(grid, rhs_fun)

    # populate spectral version of RHS
    FFT(rhs_spec, rhs_phys)

    # initialise boundary condition
    BC_neu_phys = PhysicalField(grid, BC_neu_fun)
    BC_neu_spec = SpectralField(grid)
    FFT(BC_neu_spec, BC_neu_phys)
    BC_neu = (BC_neu_spec[1, :, :], BC_neu_spec[1, :, :])

    # find solution
    solve!(ϕ_spec, laplace, rhs_spec, BC_neu)
    IFFT(ϕ_phys, ϕ_spec)

    # modify solution with offset (arbitrary when using Neumann BCs)
    for i in 1:Nt
        ϕ_phys[:, :, i] = ϕ_phys[:, :, i] .- (ϕ_phys[1, 1, i] - ϕ_sol[1, 1, i])
    end

    @test ϕ_phys ≈ ϕ_sol
end

@testset "Neumann homogeneous BC with no-slip   " begin
    # initialise constants
    β = 1.0
    Ny = 64; Nz = 64; Nt = 64

    # chebyshev points and differentiation matrix
    D2 = chebdiff(Ny); DD2 = chebddiff(Ny)
    y = chebpts(Ny)

    # initialise grid
    grid = Grid(y, Nz, Nt, D2, DD2, rand(Ny), β, 0.0)

    # initialise Laplace operators
    laplace = Laplace(Ny, Nz, β, DD2, D2, noslip=true)

    # initialise FFT plans
    FFT = FFTPlan!(grid, flags = FFTW.ESTIMATE)
    IFFT = IFFTPlan!(grid, flags = FFTW.ESTIMATE)

    # initialise functions
    sol_fun(y, z, t) = (sin(π*y)^2)*exp(cos(z))*atan(sin(t))
    rhs_fun(y, z, t) = (2*(π^2)*cos(2*π*y) + (sin(π*y)^2)*((sin(z)^2) - cos(z)))*exp(cos(z))*atan(sin(t))

    # initialise solution field
    ϕ_spec = SpectralField(grid)
    ϕ_phys = PhysicalField(grid)

    # initialise RHS field
    rhs_spec = SpectralField(grid)
    rhs_phys = PhysicalField(grid, rhs_fun)

    # populate spectral version of RHS
    FFT(rhs_spec, rhs_phys)

    # find solution
    solve!(ϕ_spec, laplace, rhs_spec, true)
    IFFT(ϕ_phys, ϕ_spec)

    @test ϕ_phys ≈ PhysicalField(grid, sol_fun)
end
