@testset "Laplace operator initialisation       " begin
    # generate random numbers
    Ny = rand(3:50)
    Nz = rand(3:50)
    β = abs(randn())
    # D = chebdiff(Ny)
    # DD = chebddiff(Ny)
    DD = DiffMatrix(range(1; stop=-1, length=Ny), 3, 2)
    D = DiffMatrix(range(1; stop=-1, length=Ny), 3, 1)
    rinner = rand(2:(Ny - 1))

    # initialise operators
    laplace_dirichlet = Laplace(Ny, Nz, β, DD)
    laplace_neumann = Laplace(Ny, Nz, β, DD, D)

    # correct size
    @test size(laplace_dirichlet.lus) == ((Nz >> 1) + 1,)
    @test size(laplace_neumann.lus) == ((Nz >> 1) + 1,)
end

@testset "Dirichlet homogeneous BC solution     " begin
    # initialise constants
    β = 1.0; ω = 1.0
    Ny = 64; Nz = 64; Nt = 64

    # chebyshev points and differentiation matrix
    y = chebpts(Ny)
    # ! chebyshev differentiation matrices don't work with lapack keyword argument !
    # D2 = chebdiff(Ny); DD2 = chebddiff(Ny)
    D2 = DiffMatrix(y, 3, 1); DD2 = DiffMatrix(y, 3, 2)

    # initialise grid
    grid = Grid(y, Nz, Nt, D2, DD2, rand(Ny), β, ω)

    # initialise Laplace operators
    # laplace = Laplace(Ny, Nz, β, DD2, :banded)
    laplace = Laplace(Ny, Nz, β, DD2, :lapack)

    # initialise FFT plans
    FFT = FFTPlan!(grid; flags=ESTIMATE)
    IFFT = IFFTPlan!(grid; flags=ESTIMATE)

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
    β = 1.0; ω = 1.0
    Ny = 64; Nz = 64; Nt = 64

    # chebyshev points and differentiation matrix
    y = chebpts(Ny)
    # D2 = chebdiff(Ny); DD2 = chebddiff(Ny)
    D2 = DiffMatrix(y, 5, 1); DD2 = DiffMatrix(y, 5, 2)

    # initialise grid
    grid = Grid(y, Nz, Nt, D2, DD2, rand(Ny), β, ω)

    # initialise Laplace operators
    # laplace = Laplace(Ny, Nz, β, DD2, D2, :banded)
    laplace = Laplace(Ny, Nz, β, DD2, D2, :lapack)

    # initialise FFT plans
    FFT = FFTPlan!(grid, flags=ESTIMATE)
    IFFT = IFFTPlan!(grid, flags=ESTIMATE)

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
    β = 1.0; ω = 1.0
    Ny = 64; Nz = 64; Nt = 64

    # chebyshev points and differentiation matrix
    y = chebpts(Ny)
    # D2 = chebdiff(Ny); DD2 = chebddiff(Ny)
    D2 = DiffMatrix(y, 3, 1); DD2 = DiffMatrix(y, 3, 2)

    # initialise grid
    grid = Grid(y, Nz, Nt, D2, DD2, rand(Ny), β, ω)

    # initialise Laplace operators
    # laplace = Laplace(Ny, Nz, β, DD2, :banded)
    laplace = Laplace(Ny, Nz, β, DD2, :lapack)

    # initialise FFT plans
    FFT = FFTPlan!(grid, flags=ESTIMATE)
    IFFT = IFFTPlan!(grid, flags=ESTIMATE)

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
    β = 1.0; ω = 1.0
    Ny = 64; Nz = 64; Nt = 64

    # chebyshev points and differentiation matrix
    y = chebpts(Ny)
    # D2 = chebdiff(Ny); DD2 = chebddiff(Ny)
    D2 = DiffMatrix(y, 5, 1); DD2 = DiffMatrix(y, 5, 2)

    # initialise grid
    grid = Grid(y, Nz, Nt, D2, DD2, rand(Ny), β, ω)

    # initialise Laplace operators
    # laplace = Laplace(Ny, Nz, β, DD2, D2, :banded)
    laplace = Laplace(Ny, Nz, β, DD2, D2, :lapack)

    # initialise FFT plans
    FFT = FFTPlan!(grid, flags=ESTIMATE)
    IFFT = IFFTPlan!(grid, flags=ESTIMATE)

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
