abstract type AbstractSingleRecurrentCell{TS} <: AbstractRecurrentCell end
abstract type AbstractDoubleRecurrentCell{TS, TM} <: AbstractRecurrentCell end

function multi_inits(rng::AbstractRNG, inits, args...)
    weights = vcat(
        [init_rnn_weight(rng, init, args...)
         for init in inits]...
    )
    return weights
end

function multi_bias(rng::AbstractRNG, inits, args...)
    biases = vcat(
        [init_rnn_bias(rng, init, args...)
         for init in inits]...
    )
    return biases
end

### single return forward
function (rcell::AbstractSingleRecurrentCell{False})(inp::AbstractMatrix,
        ps, st::NamedTuple)
    rng = replicate(st.rng)
    state = init_rnn_hidden_state(rng, rcell, inp)
    return rcell((inp, (state,)), ps, merge(st, (; rng)))
end

function (rcell::AbstractSingleRecurrentCell{True})(inp::AbstractMatrix,
        ps, st::NamedTuple)
    state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    return rcell((inp, (state,)), ps, st)
end

### double return forward
function (rcell::AbstractDoubleRecurrentCell{False, False})(inp::AbstractMatrix,
        ps, st::NamedTuple)
    rng = replicate(st.rng)
    state = init_rnn_hidden_state(rng, rcell, inp)
    c_state = init_rnn_hidden_state(rng, rcell, inp)
    return rcell((inp, (state, c_state)), ps, merge(st, (; rng)))
end

function (rcell::AbstractDoubleRecurrentCell{True, False})(inp::AbstractMatrix,
        ps, st::NamedTuple)
    rng = replicate(st.rng)
    state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    c_state = init_rnn_hidden_state(rng, rcell, inp)
    return rcell((inp, (state, c_state)), ps, merge(st, (; rng)))
end

function (rcell::AbstractDoubleRecurrentCell{False, True})(inp::AbstractMatrix,
        ps, st::NamedTuple)
    rng = replicate(st.rng)
    state = init_rnn_hidden_state(rng, rcell, inp)
    c_state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    return rcell((inp, (state, c_state)), ps, merge(st, (; rng)))
end

function (rcell::AbstractDoubleRecurrentCell{True, True})(inp::AbstractMatrix,
        ps, st::NamedTuple)
    state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    c_state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    return rcell((inp, (state, c_state)), ps, st)
end

statelength(::AbstractSingleRecurrentCell) = 1
function initialstates(rng::AbstractRNG, ::AbstractSingleRecurrentCell)
    (rng=Utils.sample_replicate(rng),)
end

statelength(::AbstractDoubleRecurrentCell) = 1
function initialstates(rng::AbstractRNG, ::AbstractDoubleRecurrentCell)
    (rng=Utils.sample_replicate(rng),)
end

function multi_initialparameters(rng::AbstractRNG, rnn::AbstractSingleRecurrentCell)
    weight_ih = multi_inits(
        rng, rnn.init_weight, rnn.out_dims, (rnn.out_dims, rnn.in_dims))
    weight_hh = multi_inits(rng, rnn.init_recurrent_weight, rnn.out_dims,
        (rnn.out_dims, rnn.out_dims))
    ps = (; weight_ih, weight_hh)
    if has_bias(rnn)
        bias_ih = multi_bias(rng, rnn.init_bias, rnn.out_dims, rnn.out_dims)
        bias_hh = multi_bias(
            rng, rnn.init_recurrent_bias, rnn.out_dims, rnn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    has_train_state(rnn) &&
        (ps = merge(ps, (hidden_state=rnn.init_state(rng, rnn.out_dims),)))
    return ps
end

function multi_initialparameters(rng::AbstractRNG, rnn::AbstractDoubleRecurrentCell)
    weight_ih = multi_inits(
        rng, rnn.init_weight, rnn.out_dims, (rnn.out_dims, rnn.in_dims))
    weight_hh = multi_inits(rng, rnn.init_recurrent_weight, rnn.out_dims,
        (rnn.out_dims, rnn.out_dims))
    ps = (; weight_ih, weight_hh)
    if has_bias(rnn)
        bias_ih = multi_bias(rng, rnn.init_bias, rnn.out_dims, rnn.out_dims)
        bias_hh = multi_bias(
            rng, rnn.init_recurrent_bias, rnn.out_dims, rnn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    has_train_state(rnn) &&
        (ps = merge(ps, (hidden_state=rnn.init_state(rng, rnn.out_dims),)))
    known(rnn.train_memory) &&
        (ps = merge(ps, (memory=rnn.init_memory(rng, rnn.out_dims),)))
    return ps
end

function single_initialparameters(rng::AbstractRNG, rnn::AbstractSingleRecurrentCell)
    weight_ih = init_rnn_weight(
        rng, rnn.init_weight, rnn.out_dims, (rnn.out_dims, rnn.in_dims))
    weight_hh = init_rnn_weight(
        rng, rnn.init_recurrent_weight, rnn.out_dims,
        (rnn.out_dims, rnn.out_dims))
    ps = (; weight_ih, weight_hh)
    if has_bias(rnn)
        bias_ih = init_rnn_bias(rng, rnn.init_bias, rnn.out_dims, rnn.out_dims)
        bias_hh = init_rnn_bias(rng, rnn.init_recurrent_bias, rnn.out_dims, rnn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    has_train_state(rnn) &&
        (ps = merge(ps, (hidden_state=rnn.init_state(rng, rnn.out_dims),)))
    return ps
end

bias_safe_multigate(::Nothing, ::Val{N}) where {N} = ntuple(_ -> nothing, N)
bias_safe_multigate(bias, v::Val{N}) where {N} = multigate(bias, v)
