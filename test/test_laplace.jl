# This file contains the test for the custom (discrete) Laplace operator type.

@testset "Laplace operator tests" begin
    # generate random numbers
    Nz = rand(0:50)
    Ny = rand(2:50)
    β = abs(randn())
    rinner = rand(2:(Ny - 1))

    # initialise operators
    laplace_dirichlet = Laplace(Nz, Ny, β, :Dirichlet)
    laplace_neumann = Laplace(Nz, Ny, β, :Neumann)

    # check error thrown for wrong boundary condition
    @test_throws ArgumentError Laplace(Nz, Ny, β, :dunno)

    # correct size
    @test size(laplace_dirichlet.lus) == (Nz + 1, )
    @test size(laplace_neumann.lus) == (Nz + 1, )
    @test size(laplace_dirichlet.lus[1]) == (Ny, Ny)
    @test size(laplace_neumann.lus[1]) == (Ny, Ny)

    # boundary conditions applied correctly
    eye1 = zeros(Ny); eye2 = copy(eye1)
    eye1[1] = 1.0; eye2[end] = 1.0
    laplace_dirichlet_recon = laplace_dirichlet.lus[1].L*laplace_dirichlet.lus[1].U
    laplace_neumann_recon = laplace_neumann.lus[1].L*laplace_neumann.lus[1].U
    @test laplace_dirichlet_recon[1, :] == eye1
    @test laplace_dirichlet_recon[end, :] == eye2
    @test laplace_neumann_recon[1, :] ≈ cheb_single_diffmat(Ny, 1)
    @test laplace_neumann_recon[end, :] ≈ cheb_single_diffmat(Ny, Ny)

    # are the inner (non-boundary condition values correct)
    double_diffmat = cheb_double_diffmat(Ny)
    @test laplace_dirichlet_recon[rinner, :] ≈ double_diffmat[rinner, :]
    @test laplace_neumann_recon[rinner, :] ≈ double_diffmat[rinner, :]

    # can a correct solution be found
    # To test this I need to:
    #   - a solution that has an interesting RHS (MIT github repo?)
    #   - a way to sample the function with Cheybyshev points in y
    #   - a scalar field type would make this implementation easier
end