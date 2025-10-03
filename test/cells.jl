@testitem "Cells" setup=[SharedTestSetup, RecurrentLayersSetup] tags=[
    :recurrent_layers
] begin
    rng = StableRNG(12345)

    for (mode, A, dev, on_gpu) in MODES
        @testset "$mode" begin
            for (name, build_cell, knobs) in RECURRENT_CELLS
                @testset "Cell: $name" begin
                    for opts in Iterators.product(((true, false) for _ in knobs)...)
                        kw = Dict(knobs .=> opts)
                        cell = build_cell(; kw...)
                        ps, st = dev(Lux.setup(rng, cell))

                        # String-keyed knobs for logging / debugging
                        cell_knobs = Dict{String, Any}(String(k) => v for (k, v) in kw)

                        # IMPORTANT: wrap in a string literal with interpolation
                        @testset "$(format_knobs(kw))" begin
                            # Read optional knobs safely
                            train_state = get(kw, :train_state, false)

                            for x_size in ((3, 2), (3,))
                                x = A(randn(rng, Float32, x_size...))

                                (y, carry), st2 = Lux.apply(cell, x, ps, st)
                                @jet cell(x, ps, st)
                                @jet cell((x, carry), ps, st)

                                if train_state
                                    @test hasproperty(ps, :hidden_state)
                                else
                                    @test !hasproperty(ps, :hidden_state)
                                end

                                @test_gradients(loss_loop,
                                    cell,
                                    x,
                                    ps,
                                    st;
                                    atol=1e-3,
                                    rtol=1e-3,)
                            end
                        end
                    end
                end
            end
        end
    end
end
