function multi_inits(rng::AbstractRNG, inits, first_dim::IntegerType, second_dim::IntegerType)
    weights = vcat(
        [
            init_rnn_weight(
                rng, init, first_dim, (first_dim, second_dim)
            )
            for init in inits
        ]...
    )
    return weights
end

function multi_bias(rng::AbstractRNG, inits, first_dim::IntegerType, second_dim::IntegerType)
    biases = vcat(
        [
            init_rnn_bias(rng, init, first_dim, second_dim)
            for init in inits
        ]...
    )
    return biases
end