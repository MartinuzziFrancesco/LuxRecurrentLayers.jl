using Test, StableRNGs, Lux, LuxTestUtils


# build a small “registry” of the different cells and their extra kwargs
const RECURRENT_CELLS = [
    (:AntisymmetricRNNCell,
        cell->AntisymmetricRNNCell(3=>5; cell...),
        [:use_bias, :train_state]),
    (:ATRCell,
        cell->ATRCell(3=>5; cell...),
        [:use_bias, :train_state]),

]

@testset "All recurrent‐layer basics" begin
    rng = StableRNG(12345)
    for (mode, A, dev, on_gpu) in MODES
        @testset "$mode" for (name, build_cell, knobs) in RECURRENT_CELLS
            for opts in Iterators.product((true,false) for _ in knobs)
               kw = Dict(knobs .=> opts)
               cell = build_cell(; kw...)
               ps, st = dev(Lux.setup(rng, cell))

                for x_size in ((3,2), (3,))
                    x = A(randn(rng, Float32, x_size...))
                    (y, carry), st2 = Lux.apply(cell, x, ps, st)
                    @jet cell(x, ps, st)
                    @jet cell((x,carry),ps, st)

                    if kw[:train_state]
                        @test hasproperty(ps, :hidden_state)
                    else
                        @test !hasproperty(ps, :hidden_state)
                    end
                    @test_gradients(loss_loop, cell, x, ps, st; atol=1e-3, rtol=1e-3)
                end
            end
        end
    end
end