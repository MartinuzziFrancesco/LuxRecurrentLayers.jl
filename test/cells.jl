@testitem "Cells" setup=[SharedTestSetup, RecurrentLayersSetup] tags=[
    :recurrent_layers
] begin
    rng = StableRNG(12345)

    for (mode, A, dev, on_gpu) in MODES
        @testset "$mode" begin
            for (name, build_cell, knobs) in RECURRENT_CELLS
                @testset "Cell: $name" begin
                    for kw in knob_settings(knobs)
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

@testitem "coRNN and UnICORNN reference recurrences" setup=[
    SharedTestSetup, RecurrentLayersSetup
] tags=[:recurrent_layers] begin
    rng = StableRNG(2021)

    @testset "coRNN" begin
        cell = coRNNCell(2 => 3)
        @test cell.dt == 0.1f0
        @test cell.gamma == 1.0f0
        @test cell.epsilon == 1.0f0

        ps, st = Lux.setup(rng, cell)
        x = Float32[0.2 -0.1; 0.4 0.3]
        h = Float32[0.1 -0.2; 0.3 0.4; -0.5 0.6]
        z = Float32[-0.2 0.1; 0.5 -0.4; 0.3 0.2]
        (y, (new_h, new_z)), _ = Lux.apply(cell, (x, (h, z)), ps, st)

        candidate = tanh.(ps.weight_ih * x .+ ps.bias_ih .+
                          ps.weight_hh * h .+ ps.bias_hh .+
                          ps.weight_ch * z .+ ps.bias_ch)
        expected_z = z .+ cell.dt .* (candidate .- cell.gamma .* h .-
                                      cell.epsilon .* z)
        expected_h = h .+ cell.dt .* expected_z
        @test new_z ≈ expected_z atol=1.0f-6
        @test new_h ≈ expected_h atol=1.0f-6
        @test y ≈ expected_h atol=1.0f-6
    end

    @testset "UnICORNN" begin
        cell = UnICORNNCell(2 => 3)
        @test cell.dt == 0.1f0
        @test cell.alpha == 0.0f0
        @test !LuxRecurrentLayers.has_recurrent_bias(cell)

        ps, st = Lux.setup(rng, cell)
        @test size(ps.weight_hh) == (3,)
        @test size(ps.weight_ch) == (3,)
        @test all(0.0f0 .<= ps.weight_hh .<= 1.0f0)
        @test all(-0.1f0 .<= ps.weight_ch .<= 0.1f0)

        x = Float32[0.2 -0.1; 0.4 0.3]
        h = Float32[0.1 -0.2; 0.3 0.4; -0.5 0.6]
        z = Float32[-0.2 0.1; 0.5 -0.4; 0.3 0.2]
        (y, (new_h, new_z)), _ = Lux.apply(cell, (x, (h, z)), ps, st)

        step = cell.dt ./ (1 .+ exp.(-ps.weight_ch))
        candidate = tanh.(ps.weight_ih * x .+ ps.bias_ih .+
                          ps.weight_hh .* h)
        expected_z = z .- step .* (candidate .+ cell.alpha .* h)
        expected_h = h .+ step .* expected_z
        @test new_z ≈ expected_z atol=1.0f-6
        @test new_h ≈ expected_h atol=1.0f-6
        @test y ≈ expected_h atol=1.0f-6

        biased = UnICORNNCell(2 => 3; use_recurrent_bias=true)
        ps_biased, st_biased = Lux.setup(rng, biased)
        (_, (new_h_biased, new_z_biased)), _ =
            Lux.apply(biased, (x, (h, z)), ps_biased, st_biased)
        step_biased = biased.dt ./ (1 .+ exp.(-ps_biased.weight_ch))
        candidate_biased = tanh.(ps_biased.weight_ih * x .+ ps_biased.bias_ih .+
                                 ps_biased.weight_hh .* h .+ ps_biased.bias_hh)
        expected_z_biased =
            z .- step_biased .* (candidate_biased .+ biased.alpha .* h)
        expected_h_biased = h .+ step_biased .* expected_z_biased
        @test new_z_biased ≈ expected_z_biased atol=1.0f-6
        @test new_h_biased ≈ expected_h_biased atol=1.0f-6
    end
end
