using BSON
using GRAPES
using GRAPES.Dates
using GRAPES.FFTW
using GRAPES.Flux
using GRAPES.GraphNeuralNetworks
using GRAPES.Graphs
using GRAPES.SeisIO
using GRAPES.SparseArrays
using GRAPES.Statistics
using LinearAlgebra: issymmetric
using Test

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
    Z_idx = findall(endswith.(S.id, "Z"))
    Z = S[Z_idx]
    Z_lon = [Z.loc[ii].lon for ii in 1:Z.n]
    Z_lat = [Z.loc[ii].lat for ii in 1:Z.n]

    # parameters for graph creation 
    rawT = 4.0 # seconds 
    predictT = 200.0 # seconds 
    sample_time = DateTime(2019, 7, 6, 3, 19, 53) # Ridgecrest origin time 
    ridge_loc = EQLoc(lat=35.77, lon = -117.599, dep=8.0) # Ridgecrest location 

    # convert data in S to float32 for inference 
    S.x = map(x -> convert.(Float32, x), S.x)

    @testset "Graph generation" begin 

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
        # create a new model from scratch 
        encode_size = 128 
        new_model = GRAPES_model((400, 3, 1, 100), 1, encode_size = encode_size)

        # create graph to model output 
        g, distance_from_earthquake, lon, lat = generate_graph(S, rawT, predictT, ridge_loc, sample_time)
        
        # perform inference 
        gout = new_model(g)
        @test length(gout.ndata.x) == length(gout.gdata.u)

        # preprocessing 
        preprocess_out = new_model.preprocess(g.ndata.x)
        @test size(preprocess_out, 2) == g.num_nodes   

        # get pga from graph 
        pga = GRAPES.getpga(g.ndata.x)
        @test size(pga) == (1, g.num_nodes)

        # encoder 
        x = vcat(preprocess_out, pga)
        x = new_model.encoder(x)
        @test size(x) == (encode_size, g.num_nodes)

        # graph network 
        x = new_model.graph_conv(g, x, g.graph[3])
        @test size(x) == (encode_size, g.num_nodes)

        # decoder network 
        x = new_model.decoder(x)
        @test size(x) == (1, g.num_nodes)

        # test that trained model loads correctly 
        model_path = "../resources/GRAPES-model.bson"
        BSON.@load model_path model 
        @test isa(model, GRAPES_model) 

        # test inference using GNNGraph input 
        gout = model(g)
        @test isa(gout, GNNGraph)

        # test inference using explicit input of graph, x, and e (e = edge distance) 
        x = model(g, g.ndata.x, g.graph[3])
        @test isa(x, Matrix)
        @test size(x) == (1, g.num_nodes)
    end

    @testset "Filtering" begin

        # filter with IIR method of Kunugi, 2013 
        S_kunugi = deepcopy(S)
        kunugi_filt!(S_kunugi)

        # check that filtering did in fact occur 
        n = length(S.x[1])
        freq = rfftfreq(n, S.fs[1])
        lowfreq = findall(freq .< 0.2)
        highfreq = findall(freq .> 2.0)
        Xfft = rfft(S.x[1])
        Kfft = rfft(S_kunugi.x[1])
        Xspec = real.(Xfft .* conj(Xfft))
        Kspec = real.(Kfft .* conj(Kfft))
        @test all(Kspec[lowfreq] .< Xspec[lowfreq]) 
        @test all(Kspec[highfreq] .< Xspec[highfreq])

        # test bandpass filter of Karim & Yamazaki, 2002 
        JMA_filt = JMA_bandpass_filter(n, S.fs[1])
        JMAfft = Xfft .* JMA_filt
        JMAspec = real.(JMAfft .* conj(JMAfft))
        @test all(JMAspec[lowfreq] .< Xspec[lowfreq]) 
        @test all(JMAspec[highfreq] .< Xspec[highfreq])

        # test that kunugi_filt and JMA_bandpass_filter give similar results 
        @test isapprox(median(JMAspec .- Kspec) + 1.0, 1.0, rtol=1e-2)
    end

    @testset "Travel time" begin 
    
        # use JMA travel times curves for Ridgecrest 
        p_travel_time = JMA_travel_time(ridge_loc, Z, "p")
        s_travel_time = JMA_travel_time(ridge_loc, Z, "s")
        @test all(p_travel_time .< s_travel_time)

        # travel time fails for far away or deep quake 
        farEQ = EQLoc(lat=-90.0, lon=117.0, dep=10.0)
        deepEQ = EQLoc(lat=34.0, lon=-117.0, dep=2000.0)
        @test_throws BoundsError JMA_travel_time(farEQ, Z, "p")
        @test_throws BoundsError JMA_travel_time(deepEQ, Z, "p")

        # travel-time for a single EQLoc and GeoLoc 
        single_p_travel_time = JMA_travel_time(ridge_loc, Z.loc[1], "p")
        single_s_travel_time = JMA_travel_time(ridge_loc, Z.loc[1], "s")
        @test isa(single_p_travel_time, Real)
        @test isa(single_s_travel_time, Real)
        @test single_p_travel_time < single_s_travel_time
        @test isapprox(single_s_travel_time / single_p_travel_time, sqrt(3.0), rtol=1e-2)

        # travel time using distance from earthquake 
        g, distance_from_earthquake, sort_lon, sort_lat = generate_graph(S, rawT, predictT, ridge_loc, sample_time)
        distance_p_travel_time = JMA_travel_time(distance_from_earthquake, ridge_loc.dep, "p") 
        distance_s_travel_time = JMA_travel_time(distance_from_earthquake, ridge_loc.dep, "s")
        @test all(isapprox.(distance_p_travel_time, sort(p_travel_time)))
        @test all(isapprox.(distance_s_travel_time, sort(s_travel_time)))

        # single AK135 travel-time 
        single_ak135_p_travel_time = predictTT(ridge_loc, Z.loc[1], "p")
        single_ak135_s_travel_time = predictTT(ridge_loc, Z.loc[1], "s")
        @test single_ak135_p_travel_time < single_ak135_s_travel_time
        @test isapprox(single_ak135_s_travel_time / single_ak135_p_travel_time, sqrt(3.0), rtol=1e-2)

        # AK135 travel-time 
        ak135_p_travel_time = predictTT(ridge_loc, Z, "p")
        ak135_s_travel_time = predictTT(ridge_loc, Z, "s")
        @test all(ak135_p_travel_time .< ak135_s_travel_time) 

        # AK135 travel time from lat/lon 
        explicit_p_travel_time = predictTT(ridge_loc.lon, ridge_loc.lat, ridge_loc.dep, Z_lon, Z_lat, "p")
        explicit_s_travel_time = predictTT(ridge_loc.lon, ridge_loc.lat, ridge_loc.dep, Z_lon, Z_lat, "s")
        @test all(explicit_p_travel_time .< explicit_s_travel_time)
        @test all(isapprox.(explicit_p_travel_time, ak135_p_travel_time))
        @test all(isapprox.(explicit_s_travel_time, ak135_s_travel_time))
    end

    @testset "Distances" begin 

        # check distance function 
        dist = distance(Z_lat, Z_lon)
        @test size(dist) == (length(Z_lat), length(Z_lat))
        @test issymmetric(dist)
        @test minimum(dist) == 0.0 # in meters 
        @test maximum(dist) > 1000.0 # in meters

        # don't use elevation 
        @test dist == distance(Z_lat, Z_lon, 10000.0 .* rand(eltype(Z_lat), size(Z_lat))) 

        # distance using Z.loc 
        @test isapprox(dist, distance(Z.loc), rtol=1e-3)

        # distance using EQ location 
        eq_dist = distance(Z_lat, Z_lon, ridge_loc)
        @test size(eq_dist) == size(Z_lat)

        # error throwing for latitude
        @test_throws ArgumentError distance(fill(91.0, length(Z_lat)), Z_lon) 

        # error throwing for different sized lat/lon 
        @test_throws ArgumentError distance(Z_lat[1:end-1], Z_lon)

        # check inputs and scaling with distance for kanno_2006_amplitude 
        k_amp_float = kanno_2006_amplitude(10, 7.0)
        @test isa(k_amp_float, Real)
        k_amp_vec = kanno_2006_amplitude(10.0 .* rand(10), 7.0)
        @test isa(k_amp_vec, Vector)
        k_amp_range = kanno_2006_amplitude(1:100, 7.0)
        @test isa(k_amp_range, Vector)
        @test all(diff(k_amp_range) .< 0.0) # check that amplitude decreases with distance
        k_mag_range = kanno_2006_amplitude(10.0, 1:7)
        @test isa(k_mag_range, Vector)
        @test all(diff(k_mag_range) .> 0.0) # check that amplitude increases with magnitude

        # test adjacency creation with threshold 
        thresh_dist = 10000.0 # meters
        adj = adjacency(dist, thresh_dist)
        @test isa(adj, SparseMatrixCSC)
        @test length(adj.nzval) < length(dist)
        @test issymmetric(adj)

        # test adjacency with all but self connections 
        adj_all = adjacency(dist)
        @test isa(adj_all, SparseMatrixCSC)
        @test length(adj_all.nzval) == adj_all.m * (adj_all.n - 1) 
        @test maximum(adj_all.nzval) == 1.0
        
        # test adjacency from graph 
        g, distance_from_earthquake, lon, lat = generate_graph(S, rawT, predictT, ridge_loc, sample_time)
        el = Edge.([(src,dst) for (src,dst) in zip(g.graph[1],g.graph[2])])
        graph = SimpleDiGraph(el)
        adj_graph = adjacency(graph, dist)
        @test isa(adj_graph, SparseMatrixCSC)
        @test length(adj_graph.nzval) < length(dist)
        @test !issymmetric(adj_graph)

        # test knn adjacency
        k = 20 
        knn_adj = knn_adjacency(dist, k)
        @test isa(knn_adj, SparseMatrixCSC)
        @test length(knn_adj.nzval) == size(dist,1) * k
        # check huge k 
        huge_k = size(knn_adj,1) * 10 
        @test_throws AssertionError knn_adjacency(dist, huge_k)

        # test navigation from Kleinberg, 2000 
        α = 2.0 
        q = 1
        g_knn = SimpleDiGraph(knn_adj)
        g_small_world = kleinberg_navigation(g_knn, α, q)
        @test isa(g_small_world, SimpleDiGraph)
        @test g_knn.ne < g_small_world.ne

        # test reverse_distance 
        dist_rev = reverse_distance(dist)
        @test typeof(dist) == typeof(dist_rev)
        @test issymmetric(dist_rev)
        @test maximum(dist_rev) == 1.0
        @test minimum(dist_rev) == 0.0
    end

    @testset "Utils" begin 
        # test smoothing 
        x = repeat([0.0, 1.0], 10000)
        x_smooth = smooth(x, 10)
        @test size(x) == size(x_smooth)
        @test eltype(x) == eltype(x_smooth) 
        @test isapprox(mean(x_smooth), 0.5, rtol=1e-2)
        # in-place smoothing 
        smooth!(x, 10)
        @test x == x_smooth 

        # test moving_abs_maximum 
        x = rand(Float32, 1000000)
        x .-= mean(x) # zero mean 
        x .*= 2 # vary from -1 to 1 
        xmax = moving_abs_maximum(x, 100)
        @test size(x) == size(xmax)
        @test eltype(x) == eltype(xmax)
        @test isapprox(mean(xmax), 0.5, rtol = 1e-2) # xmax varies from in (0,1)

        # test moving_abs_maximum on SeisChannel 
        C = deepcopy(S[1])
        Cout = moving_abs_maximum(C, 100)
        @test size(C.x) == size(Cout.x)
        @test eltype(C.x) == eltype(Cout.x)
        @test mean(C.x) < mean(Cout.x)

        # test moving_abs_maximum on SeisData
        Sout = moving_abs_maximum(S, 100)
        @test length.(S.x) == length.(Sout.x)
        @test eltype.(S.x) == eltype.(Sout.x)
        @test all(mean.(S.x) .< mean.(Sout.x)) 

        # test envelope 
        x_env = envelope(x)
        @test size(x) == size(x_env)
        @test eltype(x) == eltype(x_env)
        @test all(x_env .+ 5 * Flux.epseltype(x) .> x)


    end

    @testset "I/O" begin 
        """
        io.jl implements loading of K-NET/KiK-net waveforms and JMA hypocenter files.
        JMA does not allow outside organizations/individuals to post these data,
        thus these function will remain untested. 
        """
    end
end