abstract type AbstractSingleRecurrentCell{TS} <: AbstractRecurrentCell end
abstract type AbstractDoubleRecurrentCell{TS, TM} <: AbstractRecurrentCell end

function multi_inits(args...)
    weights = vcat(
        [init_rnn_weight(args...)
         for init in inits]...
    )
    return weights
end

function multi_bias(args...)
    biases = vcat(
        [init_rnn_bias(args...)
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
