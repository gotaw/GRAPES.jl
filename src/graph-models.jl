export GRAPES_model

# type for GRAPES models. Layers and parameters are fields 
struct GRAPES_model
    getpga
    preprocess                          
    encoder
    graph_conv 
    decoder
end

 # make GRAPES_model Flux compatible
Flux.@functor GRAPES_model               

"""

  GRAPES_model(inputsize, outputlength)

Graph Neural Network model for predicting PGA from raw data with 4 steps:

    1. Pre-process raw acceleration records individually with 1D convolutions
    2. Encode downsampled acceleration data
    3. Share data between stations with graph convolutions
    4. Decode graph data into PGA prediction

# Arguments
- `inputsize::Tuple`: Size of input raw data (sample length x channels x 1 x stations)
- `outputlength::Integer`: Length of the output from model (PGA features)

# Optional
- `preprocess`: Function or chain of operations to pre-process data along channels 
- `encode_size::Integer`: Size of hidden layers in encoder and Graph Convolution 
- `N_encoder_layers::Integer`: Number of encoder layers 
- `N_graph_layers::Integer`: Number of graph convolution layers 
- `aggr::Function`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`). Default `max`. 
- `dropout_p::Real`: Probability to drop neurons in dense layers. Defaults to no dropout. 
"""
function GRAPES_model(
    inputsize::Tuple, 
    outputlength::Integer;
    preprocess=preprocess,
    encode_size::Integer=256,
    N_encoder_layers::Integer=2,
    N_graph_layers::Integer=3,
    aggr::Function=max,
    dropout_p::Real=0.0,
)  

    # calculation of output dimension size 
    pp_dims = Flux.outputsize(preprocess, inputsize)
    preprocess_out = pp_dims[1] + 1 # plus PGA

    # chain for encoder network 
    encoder = Chain(
        vcat(
            Dense(preprocess_out, encode_size, relu),
            Dropout(dropout_p),
            [[Dense(encode_size, encode_size, relu), Dropout(dropout_p)] for ii in 1:N_encoder_layers]...,
        )
    )

    # GNNchain for graph convolution 
    graph_conv = GNNChain(
        [WeightedGraphConv(encode_size=>encode_size, relu, aggr=aggr) for ii in 1:N_graph_layers]
    )

    # chain for decoder network 
    decoder = Chain(
        vcat(
            [Dense(encode_size, encode_size, relu) for ii in 1:N_encoder_layers],
            Dense(encode_size, outputlength),
        )
    )

    GRAPES_model(
        getpga, 
        preprocess,
        encoder,
        graph_conv,
        decoder,
    )
end

 # Define the forward pass
(model::GRAPES_model)(g::GNNGraph) = GNNGraph(g, ndata=(x=model(g, g.ndata.x, g.graph[3])))
function (model::GRAPES_model)(g::GNNGraph, x::AbstractArray, e::AbstractArray)

    # extract pga from model 
    pga = model.getpga(x)

    # apply preprocessing 
    x = model.preprocess(x)

    # concatenate pga
    x = vcat(x, pga)

    # encoding layers 
    x = model.encoder(x)

    # graph convolutions over stations 
    x = model.graph_conv(g, x, e)

    # decode to pga prediction 
    x = model.decoder(x)
    return x 
end

# function to extract PGA from current sample 
getpga(x) = Flux.flatten(log10.(sqrt.(maximum(sum(x .^ 2, dims=2), dims=(1,2)))))

"""

  jma_pga(x)

JMA calculation of PGA. 

Returns maximum log10 acceleration exceeded for 0.3 seconds at each station. 
"""
function jma_pga(x; fs::Real = 100.0, T::Real=0.3)
    xsum = sqrt.(sum(x .^ 2, dims=2))
    xsort = sort(abs.(xsum), dims=1, rev=true)
    third_second_ind = ceil(Int, fs * T) 
    return log10.(xsort[third_second_ind,:,1,:])
end

"""
    maxnorm12(x;  ϵ=1e-5)
Normalise `x` to mean 0 across first dimension and then maximum of standard deviation 1 across the first dimension.
`ϵ` is a small additive factor added to the denominator for numerical stability.
"""
@inline function maxnorm12(x::AbstractArray; dims=ndims(x), ϵ=Flux.ofeltype(x, 1e-5))
  μ = mean(x, dims=1)
  σ = maximum(std(x, dims=1, mean=μ, corrected=false), dims=2)
  return @. (x - μ) / (σ + ϵ)
end

# custom preprocessing chain 
# applied over stations 
preprocess = Chain(
    maxnorm12,  # normalize over channels
    Conv((3,1), 1 => 8, relu),     # convolution along channel
    Conv((3,1), 8 => 32, relu),    # convolution along channel
    BatchNorm(32),                 # Batch normalization 
    MaxPool((2,1),),               # Pooling along channel 
    Conv((3,1,), 32 => 64, relu),  # convolution along channel
    Conv((3,1,), 64 => 64, relu),  # convolution along channel
    BatchNorm(64),                 # Batch normalization
    MaxPool((2,1),),               # Pooling along channel 
    Conv((3,1,), 64 => 128, relu), # convolution along channel
    BatchNorm(128),                # Batch normalization
    MaxPool((2,1),),               # Pooling along channel 
    Conv((3,1,), 128 => 32, relu), # convolution along channel
    Conv((3,3,), 32 => 16, relu),  # convolution along both channels
    MaxPool((2,1),),               # Pooling along channel 
    Flux.flatten,                  # channel flattening
)

@doc raw"""
    WeightedGraphConv(in => out, σ=identity; aggr=+, bias=true, init=glorot_uniform)
Graph convolution layer from Reference: [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244).
Performs:
```math
\mathbf{x}_i' = W_1 \mathbf{x}_i + \square_{j \in \mathcal{N}(i)} e_{ij} W_2 \mathbf{x}_j
```
where the aggregation type is selected by `aggr` and `e_{ij}` is distance from node `i` to node `j`.
# Arguments
- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
"""
struct WeightedGraphConv{A<:AbstractMatrix, B} <: GNNLayer
    weight1::A
    weight2::A
    bias::B
    σ
    aggr
end

Flux.@functor WeightedGraphConv

function WeightedGraphConv(ch::Pair{Int,Int}, σ=identity; aggr=+,
                   init=Flux.glorot_uniform, bias::Bool=true)
    in, out = ch
    W1 = init(out, in)
    W2 = init(out, in)
    b = bias ? Flux.create_bias(W1, true, out) : false
    WeightedGraphConv(W1, W2, b, σ, aggr)
end

function (l::WeightedGraphConv)(g::GNNGraph, x::AbstractMatrix, e::AbstractVector)
    GNNGraphs.check_num_nodes(g, x)
    m = propagate(w_mul_xj, g, l.aggr, xj=x, e=e)
    x = l.σ.(l.weight1 * x .+ l.weight2 * m .+ l.bias)
    return x
end

function (l::WeightedGraphConv)(g::GNNGraph, x::AbstractMatrix)
    GNNGraphs.check_num_nodes(g, x)
    m = propagate(w_mul_xj, g, l.aggr, xj=x, e=g.graph[3])
    x = l.σ.(l.weight1 * x .+ l.weight2 * m .+ l.bias)
    return x
end

function Base.show(io::IO, l::WeightedGraphConv)
    in_channel = size(l.weight1, ndims(l.weight1))
    out_channel = size(l.weight1, ndims(l.weight1)-1)
    print(io, "WeightedGraphConv(", in_channel, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end

# for explicit evaluation of chains with edge values 
(c::GNNChain)(g::GNNGraph, x, e) = _applychain(c.layers, g, x, e)

function _applychain(layers, g::GNNGraph, x, e) 
    for l in layers
        x = _applylayer(l, g, x, e)
    end
    return x
end

_applylayer(l::GNNLayer, g::GNNGraph, x, e) = l(g, x, e)


