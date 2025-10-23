struct AdditiveIntegration end
struct MultiplicativeIntegration end

function recurrence_double_bias(integration_mode, wih::AbstractMatrix, whh::AbstractMatrix,
        inp::AbstractArray, state::AbstractArray, bih::Union{AbstractVector, Nothing},
        bhh::Union{AbstractVector, Nothing}, bmi::Union{AbstractVector, Nothing};
    activation_inp = identity, activation_state = identity)
    wih_inp_bih = fused_dense_bias_activation(activation_inp, wih, inp, bih)
    whh_state_bhh = fused_dense_bias_activation(activation_state, whh, state, bhh)

    return dense_integration(integration_mode, wih_inp_bih, whh_state_bhh, bmi)
end

function dense_integration(::AdditiveIntegration, wih_inp_bih::AbstractMatrix,
        whh_state_bhh::AbstractMatrix, bmi::Union{AbstractVector, Nothing})

    return bias_activation(identity, wih_inp_bih .+ whh_state_bhh, bmi)
end

function dense_integration(::MultiplicativeIntegration, wih_inp_bih::AbstractMatrix,
        whh_state_bhh::AbstractMatrix, bmi::Union{AbstractVector, Nothing})

    return bias_activation(identity, wih_inp_bih .* whh_state_bhh, bmi)
end
