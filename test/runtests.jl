using BSON
using Dates 
using GRAPES
using GraphNeuralNetworks
using Flux
using Test
using SeisIO
using Statistics

"""
A simple test set for GRAPES.jl. 

GRAPES was trained on data from the K-NET and KiK-net in Japan (https://www.kyoshin.bosai.go.jp/). 
Downloading data from K-NET/KiK-net requires a login and the data cannot be distributed. Instead, 
we'll use data from the M7.1 Ridgecrest earthquake to test GRAPES.

"""

# download data from M7.1 Ridgecrest earthquake for testing 
# data is stored in ../resources/Ridgecrest/waveforms 
include("download.jl")
# minimal processing to standardize sampling rates + gain 
include("process.jl")
# run model on Ridgecrest 
include("inference.jl")

@testset "GRAPES.jl" begin

    # load waveforms from Ridgecrest for testing 
    S = read_data("../resources/SCSN.seisio")

    @testset "Graph generation" begin 
        rawT = 4.0 # seconds 
        predictT = 200.0 # seconds 
        sample_time = DateTime(2019, 7, 6, 3, 19, 53) # Ridgecrest origin time 
        ridge_loc = EQLoc(lat=35.77, lon = -117.599, dep=8.0)

        # generate a graph for Ridgecrest 
        k = 10 # neighbors 
        q = 0 # no long-range connection 
        maxdist = 0.0 # no nearby stations 
        g1, distance_from_earthquake, lon, lat = generate_graph(S, rawT, predictT, ridge_loc, sample_time, k=k, q=q, maxdist=maxdist)
        # size of node features 
        @test size(g1.ndata.x) == (round(Int, S.fs[1] * rawT), 3, 1, S.n ÷ 3)
        # size of output features 
        @test size(g1.gdata.u) == (S.n ÷ 3, 1) 
        # size of input graph (includes self connection)
        @test g1.num_graphs == 1
        @test g1.num_nodes == S.n ÷ 3
        @test g1.num_edges == (k + 1) * (S.n ÷ 3) 
        @test size(g1.graph[1]) == ((k + 1) * (S.n ÷ 3),)
        @test size(g1.graph[2]) == ((k + 1) * (S.n ÷ 3),)
        @test maximum(g1.graph[3]) <= 1.0
        @test minimum(g1.graph[3]) >= 0.0
        # test sizes of distance_from_earthquake, lon, lat
        @test length(distance_from_earthquake) == length(lon) == length(lat) == g1.num_nodes
        @test all(diff(distance_from_earthquake) .> 0)

        # add a long-range connection 
        q = 1 
        g2, distance_from_earthquake, lon, lat = generate_graph(S, rawT, predictT, ridge_loc, sample_time, k=k, q=q, maxdist=maxdist)
        # number of edges is now k + q + 1 (with some randomness)
        @test g1.num_edges < g2.num_edges <= (k + q + 1) * (S.n ÷ 3)

        # test maxdist 
        q = 0 
        maxdist = -1.0 # do not add self-connection 
        g3, distance_from_earthquake, lon, lat = generate_graph(S, rawT, predictT, ridge_loc, sample_time, k=k, q=q, maxdist=maxdist)
        @test g3.num_edges == k * (S.n ÷ 3)

        # connect all edges 
        maxdist = 1e7 # 10,000 km 
        g4, distance_from_earthquake, lon, lat = generate_graph(S, rawT, predictT, ridge_loc, sample_time, k=k, q=q, maxdist=maxdist)
        @test g4.num_edges == (S.n ÷ 3) * (S.n ÷ 3 - 1) 

        # normalize input waveforms - check 
        g5, distance_from_earthquake, lon, lat = generate_graph(S, rawT, predictT, ridge_loc, sample_time, normalize=true)
        station_std = vec(maximum(std(g5.ndata.x, dims=1), dims=2))
        @test isapprox(mean(station_std), 1.0, rtol=1e-2)

        # check logpga output 
        g6, distance_from_earthquake, lon, lat = generate_graph(S, rawT, predictT, ridge_loc, sample_time, logpga=true)
        @test maximum(abs.(g6.gdata.u)) < 3.0 # log10(1000) cm/s^2 

        # test magnitude-dependent attenuation 
        q = 0 
        maxdist = -1.0 # do not add self-connection
        g7, distance_from_earthquake, lon, lat = generate_graph(S, rawT, predictT, ridge_loc, sample_time, k=k, q=q, maxdist=maxdist, magnitude=6.0)
        @test all(g7.graph[3] .< g3.graph[3])

        # test JMA-style PGA sorting 
        g8, distance_from_earthquake, lon, lat = generate_graph(S, rawT, predictT, ridge_loc, sample_time, k=k, q=q, maxdist=maxdist, jma_sort=true)
        @test all(g8.gdata.u .< g3.gdata.u)
    end

    @testset "GRAPES model" begin

        # test that model loads correctly 
        model_path = "../resources/GRAPES-model.bson"
        BSON.@load model_path model 
        @test isa(model, GRAPES_model) 

        # create a new model from scratch 
        new_model = GRAPES_model((400, 3, 1, 100), 1)
    end
    # test filtering -> apply kunugi_filt / JMA filt
    # test generating graph -> generate graph 
    # test inference  -> apply model to graph 
end