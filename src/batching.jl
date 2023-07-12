export generate_graph_batch, generate_test_batch, generate_noise_batch, read_noise, generate_graph

"""
    generate_graph_batch(seisiopath, rawT, predictT, EQ, origin_time, intensity)

Creates a batch of seismic waveform graphs for training. 

# Arguments
- `seisiopath::String`: Path to input .seisio file 
- `rawT::AbstractFloat`: Time window of raw data to make prediction, seconds 
- `predictT::AbstractFloat`: Forward prediction time, seconds.
- `EQ::EQLoc`: Location of earthquake
- `origin_time::DateTime`: Origin time of earthquake 
- `intensity::Real`: Maximum recorded JMA Intensity of earthquake 

# Optional 
-`stepT::Real`: time between samples, seconds
-`nstations::Integer`: Maximum number of stations to sample per graph 
-`k::Integer`: number of nearest neigherbors per station
-`maxdist::Real`: Connect all stations within maxdist, meters 
-`α::Real`: clustering exponent for long-range connection  
-`q::Integer`: Number of long-range connections per station
-`oversample::Real`: Controls number of samples per `stepT`
-`biased::Bool`: Bias sampling of stations closer to the epicenter  
-`batchsize::Integer`: Number of graphs per batch 
-`logpga::Bool`: Make target of graph log10 of peak ground acceleration (PGA)
-`magntiude::Real`: Earthquake magnitude for magnitude-dependent attentuation
-`filter::Bool`: Apply IIR bandpass filter to data  

# Returns 
- `graphs::Array{GNNGraph}`: Array of seismic station graphs 
- `stations::AbstractArray`: Stations in each seismic station graph 
- `starttimes::AbstractArray`: Sample start time of each seismic station graph 
"""
function generate_graph_batch(
    seisiopath::String,
    rawT::AbstractFloat,
    predictT::AbstractFloat,
    EQ::EQLoc,
    origin_time::DateTime, 
    intensity::Real;
    stepT::Real=1.0,
    nstations::Integer=100, 
    k::Integer=10,          
    maxdist::Real=20000.0,  
    α::Real=2.0,           
    q::Integer=1,
    oversample::Real=1.0,
    normalize::Bool=false, 
    biased::Bool=false,
    batchsize::Integer=128,
    logpga::Bool=false, 
    magnitude::Real=7.0,
    filter::Bool=false,
)   
    T = Float32 
    S = read_data("seisio", seisiopath)

    if filter 
        taper!(S)
        kunugi_filt!(S)
    end

    # grab E, N, and Z components 
    Eind = findall(endswith.(S.id, "E")) 
    Nind = findall(endswith.(S.id, "N")) 
    Zind = findall(endswith.(S.id, "Z")) 
    E = deepcopy(S[Eind])
    N = deepcopy(S[Nind])
    Z = deepcopy(S[Zind]) 

    if k + q >= Z.n 
        return nothing, nothing, nothing
    end

    # number of stations to samples 
    nstations = min(Z.n, nstations)

    # time axis 
    starttime = SeisIO.u2d(SeisIO.starttime(Z.t[1], Z.fs[1]) * 1e-6)
    endtime = SeisIO.u2d(SeisIO.endtime(Z.t[1],Z.fs[1]) * 1e-6)
    t = Float64((starttime - origin_time).value / 1000) : 1 / Z.fs[1] : Float64((endtime - origin_time).value / 1000)

    # distance from earthquake 
    lon = [Z.loc[kk].lon for kk in 1:Z.n]
	lat = [Z.loc[kk].lat for kk in 1:Z.n]
    el = [Z.loc[kk].el for kk in 1:Z.n]
    distance_from_earthquake = distance([lat;EQ.lat], [lon; EQ.lon])[1:end-1,end] ./ 1000
    distance_indices = sortperm(distance_from_earthquake)
    distance_from_earthquake = distance_from_earthquake[distance_indices]
    E = E[distance_indices]
    N = N[distance_indices]
    Z = Z[distance_indices]
    lon = lon[distance_indices]
    lat = lat[distance_indices]

    # predict arrival times 
    tP = JMA_travel_time(EQ, Z, "p")
    tS = JMA_travel_time(EQ, Z, "s")
    tsurf = tS ./ 0.92 
    tstart = minimum(tP)        # starttime 
    tstop = maximum(tsurf)      # stop sampling at surface wave arrival plus 5 seconds 
    tstop = min(tstop, last(t))

    # transform into 4D tensor 
    X = cat(hcat(E.x...), hcat(N.x...), hcat(Z.x...), dims=4)
    X = permutedims(X, (1,4,3,2))
    
    # output waveforms + PGV into graphs 
    all_graphs = GNNGraph[]
    stations = Vector{String}[]
    starttimes = Int[]

    # select noise windows
    p_sample = findfirst(t .>= tstart)
    last_sample = findfirst(t .>= tstop) 
    raw_sample = round(Int, S[1].fs * rawT)
    step_sample = round(Int, S[1].fs * stepT)
    pga_sample = round.(Int, S[1].fs * predictT)
    X_total_sample = size(X, 1)

    # biased sampling toward the nearest stations 
    w = softmax(1 .- (distance_from_earthquake ./ maximum(distance_from_earthquake)))

    # biased sampling to events with larger intensity  
    intensity_samples = round(Int, 2.0 ^ (intensity - 2.0)) 
    total_samples = ceil(Int, intensity_samples * oversample)
    Nsteps = length(p_sample:step_sample:last_sample - 1)
    step_weighting = -sigmoid.(range(-5,5, Nsteps)) .+ 1
    step_weighting ./= sum(step_weighting)
    step_samples = round.(Int, total_samples .* step_weighting)   

    # ensure number of samples is divisible by batchsize
    if all(iszero(step_samples))
        step_samples[1:min(batchsize, length(step_samples))] .= 1
    end
    while sum(step_samples) % batchsize != 0 
        for ii in eachindex(step_samples)
            if sum(step_samples) % batchsize == 0
                break
            end
            if step_samples[ii] != 0 
                step_samples[ii] += 1 
            end
        end
    end
        
    # get p-wave samples 
    for (ii, current_sample) in enumerate(p_sample:step_sample:last_sample - 1) 
        for jj in 1:step_samples[ii]

            # random station sampling
            if biased 
                random_station_sample = sample(1:Z.n, Weights(w), nstations, replace=false)
            else
                random_station_sample = sample(1:Z.n, nstations, replace=false)
            end
            Zn = Z[random_station_sample]
            minimum_p_sample = findfirst(t .>= minimum(tP[random_station_sample])) 

            # create adjacency graph 
            station_distance = distance(Zn.loc)
            adjK = knn_adjacency(station_distance, k)
            adjD = adjacency(station_distance, maxdist)
            adj = adjK + adjD 
            g = SimpleDiGraph(adj)

            # add shortcuts, if needed 
            if q > 0
                kleinberg_navigation!(g, α, q)
            end
            adj = adjacency(g, kanno_2006_amplitude(station_distance ./ 1000.0, magnitude))
            
            # extract raw + pga data 
            # add small random time-shift window start 
            maxshift = round(Int, 0.5 * raw_sample) 
            tshift = rand(-maxshift:maxshift)
            max_window_sample = pga_sample + max(current_sample, minimum_p_sample) + tshift + raw_sample 
            max_window_sample > X_total_sample && continue 
            Xwindow = @view X[max(current_sample, minimum_p_sample)+tshift:max_window_sample,:,:,random_station_sample]
            Xenv = envelope(Xwindow[raw_sample+1:end,:,:,:])

            # downsample PGV data 
            pga = dropdims(sqrt.(sum(maximum(Xenv .^ 2, dims=1), dims=2)), dims=(1,2,3))

            if logpga
                pga = log10.(pga)
            end

            # reshape raw to 4D array 
            raw = Xwindow[1:raw_sample,:,:,:]

            # normalize 
            if normalize
                raw = maxnorm12(raw)
            end

            G = GNNGraph(adj, ndata = (x=raw), gdata = (u=pga))
            push!(all_graphs, G)
            push!(stations, replace.(Zn.id,"..HNZ"=>""))
            push!(starttimes, current_sample + tshift)
        end
    end

    # batching 
    Ngraphs = length(all_graphs)
    Nbatches = ceil(Int, Ngraphs / batchsize)
    g = Array{GNNGraph}(undef, Nbatches)
    for ii in 1: Nbatches
        g[ii] = GRAPES.batch(all_graphs[(ii-1) * batchsize + 1 : min(ii * batchsize, Ngraphs)])
    end
    return g, stations, starttimes
end

"""
    generate_test_batch(seisiopath, rawT, predictT, EQ, origin_time)

Creates a batch of seismic waveform graphs for inference. 

# Arguments
- `seisiopath::String`: Path to input .seisio file 
- `rawT::AbstractFloat`: Time window of raw data to make prediction, seconds 
- `predictT::AbstractFloat`: Forward prediction time, seconds.
- `EQ::EQLoc`: Location of earthquake
- `origin_time::DateTime`: Origin time of earthquake 

# Optional 
-`stepT::Real`: time between samples, seconds
-`k::Integer`: number of nearest neigherbors per station
-`maxdist::Real`: Connect all stations within maxdist, meters 
-`α::Real`: clustering exponent for long-range connection  
-`q::Integer`: Number of long-range connections per station
-`logpga::Bool`: Make target of graph log10 of peak ground acceleration (PGA)
-`magntiude::Real`: Earthquake magnitude for magnitude-dependent attentuation
-`filter::Bool`: Apply IIR bandpass filter to data  

# Returns 
- `all_graphs::Array{GNNGraph}`: Array of seismic station graphs 
- `window_start_times::Vector`: Start time of each seismic station graph 
- `window_end_times::Vector`: Start time of each seismic station graph
- `distance_from_earthquake::Vector`: Stations in each seismic station graph  
- `lon::Vector`: Station longitude 
- `lat::Vector`: Station latitude 
"""
function generate_test_batch(
    seisiopath::String,
    rawT::AbstractFloat,
    predictT::AbstractFloat,
    EQ::EQLoc,
    origin_time::DateTime;
    stepT::Real=1.0,
    k::Integer=10,          # K-nearest neighbors  
    maxdist::Real=20000.0,  # meters 
    α::Real=2.0,
    q::Integer=1,
    normalize::Bool=false, 
    logpga::Bool=false,
    magnitude::Real=7.0,
    filter::Bool=false,
)   
    T = Float32 
    S = read_data("seisio", seisiopath)

    if filter 
        taper!(S)
        kunugi_filt!(S)
    end

    # grab E, N, and Z components 
    Eind = findall(endswith.(S.id, "E")) 
    Nind = findall(endswith.(S.id, "N")) 
    Zind = findall(endswith.(S.id, "Z")) 
    E = deepcopy(S[Eind])
    N = deepcopy(S[Nind])
    Z = deepcopy(S[Zind]) 

    # time axis 
    starttime = SeisIO.u2d(SeisIO.starttime(Z.t[1], Z.fs[1]) * 1e-6)
    endtime = SeisIO.u2d(SeisIO.endtime(Z.t[1],Z.fs[1]) * 1e-6)
    t = Float64((starttime - origin_time).value / 1000) : 1 / Z.fs[1] : Float64((endtime - origin_time).value / 1000)

    # distance from earthquake 
    lon = [Z.loc[kk].lon for kk in 1:Z.n]
	lat = [Z.loc[kk].lat for kk in 1:Z.n]
    el = [Z.loc[kk].el for kk in 1:Z.n]
    distance_from_earthquake = distance([lat;EQ.lat], [lon; EQ.lon])[1:end-1,end] ./ 1000
    distance_indices = sortperm(distance_from_earthquake)
    distance_from_earthquake = distance_from_earthquake[distance_indices]
    E = E[distance_indices]
    N = N[distance_indices]
    Z = Z[distance_indices]
    lon = lon[distance_indices]
    lat = lat[distance_indices]
    el = el[distance_indices]

    # create adjacency matrix 
    station_distance = distance(lat, lon)
    adjK = knn_adjacency(station_distance, k)
    adjD = adjacency(station_distance, maxdist)
    adj = adjK + adjD 
    g = SimpleDiGraph(adj)

    # add shortcuts, if needed 
    if q > 0
        kleinberg_navigation!(g, α, q)
    end
    adj = adjacency(g, kanno_2006_amplitude(station_distance ./ 1000.0, magnitude))

    # predict arrival times 
    p_arrival_time = JMA_travel_time(EQ, Z[1:1], "p")[1]
    s_arrival_time = JMA_travel_time(EQ, Z[end:end], "s")[1]
    surface_wave_arrival_time = s_arrival_time / 0.92 
    tsurf_minus_tp = max.(surface_wave_arrival_time .- p_arrival_time, predictT) # scale prediction time based on distance from epicenter
    tstart = -15.0 # start sampling at 15 seconds before origin time 
    tstop = surface_wave_arrival_time .+ 5.0 # stop sampling at surface wave arrival plus 5 seconds 
    tstop = min(tstop, last(t))

    # transform into 4D tensor 
    X = cat(hcat(E.x...), hcat(N.x...), hcat(Z.x...), dims=4)
    X = permutedims(X, (1,4,3,2))

    # select noise windows
    taper_sample = 100
    start_sample = max(findfirst(t .>= tstart + taper_sample / S.fs[1]), taper_sample + 1) 
    start_sample = findfirst(ceil.(t[start_sample]) .== t) # round start to next whole second 
    last_sample = findfirst(t .>= tstop) 
    raw_sample = round(Int, S[1].fs * rawT)
    step_sample = round(Int, S[1].fs * stepT)
    X_total_sample = size(X, 1)
    pga_sample = round.(Int, S[1].fs .* predictT)

    # output waveforms + PGV into graphs 
    all_graphs = GNNGraph[]
    window_start_times = Float64[]
        
    # get p-wave samples 
    for (ii, current_sample) in enumerate(start_sample:step_sample:last_sample - 1) 
        # extract raw + pga data 
        # add small random time-shift window start 
        max_window_sample = pga_sample + current_sample + raw_sample + taper_sample
        max_window_sample > X_total_sample && continue 
        raw = X[current_sample:current_sample+raw_sample-1,:,:,:]
        Xwindow = @view X[current_sample+raw_sample+1:max_window_sample,:,:,:]

        # downsample PGV data 
        pga = dropdims(sqrt.(sum(maximum(Xwindow .^ 2, dims=1), dims=2)), dims=(1,2,3))

        if logpga
            pga = log10.(pga)
        end

        # normalize 
        if normalize
            raw = maxnorm12(raw)
        end

        push!(all_graphs, GNNGraph(adj, ndata = (x=raw,), gdata= (u=pga,)))
        push!(window_start_times, t[current_sample])
    end
    window_end_times = window_start_times .+ rawT 
    return all_graphs, window_start_times, window_end_times, distance_from_earthquake, lon, lat 
end

function generate_noise_batch(
    noise_array::AbstractArray, 
    lon::AbstractArray, # station longitudes 
    lat::AbstractArray, # station latitudes 
    inputsize::Tuple, 
    nbatches::Integer;
    nstations::Integer=100, # number of stations to sample 
    k::Integer=10,          # K-nearest neighbors  
    maxdist::Real=20000.0,  # meters 
    α::Real=2.0,
    q::Integer=1,
    normalize::Bool=false, 
    biased::Bool=false,
    batchsize::Integer=128,
    logpga::Bool=false,
    magnitude::Real=7.0,
)   
    # output waveforms + PGA into graphs 
    noise_samples = inputsize[1]
    ngraphs = nbatches * batchsize
    all_graphs = Array{GNNGraph}(undef, ngraphs)

    # get distance between all stations 
    station_distance = distance(lat, lon)
        
    # random noise sampling 
    for ii in 1:ngraphs
        # select k random stations 
        noise_random_selection = sample(1:length(noise_array), nstations, replace=false)
        event_noise = zeros(Float32, inputsize)
        for jj in eachindex(noise_random_selection)
            time_samples = size(noise_array[noise_random_selection[jj]], 1)
            sample_start = rand(1:(time_samples - noise_samples))
            event_noise[:,:,1,jj] .= noise_array[noise_random_selection[jj]][sample_start:sample_start+noise_samples-1,:]
        end

        # create adjacency graph 
        random_station = rand(1:size(station_distance, 1))
        nearest_random_stations = sortperm(station_distance[:,random_station])[1:nstations]
        sample_station_distance = station_distance[nearest_random_stations, nearest_random_stations]
        adjK = knn_adjacency(sample_station_distance, k)
        adjD = adjacency(sample_station_distance, maxdist)
        adj = adjK + adjD 
        g = SimpleDiGraph(adj)

        # add shortcuts, if needed 
        if q > 0
            kleinberg_navigation!(g, α, q)
        end
        adj = adjacency(g, kanno_2006_amplitude(sample_station_distance ./ 1000.0, magnitude))

        # downsample PGV data 
        peak_noise = 3.3 * maximum(median(envelope(event_noise), dims=1), dims=2)[:]

        if logpga
            peak_noise = log10.(peak_noise)
        end

        # normalize 
        if normalize
            event_noise = maxnorm12(event_noise)
        end

        G = GNNGraph(adj, ndata = (x=event_noise,), gdata= (u=peak_noise))
        all_graphs[ii] = G
    end

    # batching 
    g = Array{GNNGraph}(undef, nbatches)
    for ii in 1: nbatches
        g[ii] = fast_noise_batch(all_graphs[(ii-1) * batchsize + 1 : min(ii * batchsize, ngraphs)], nstations)
        all_graphs[(ii-1) * batchsize + 1 : min(ii * batchsize, ngraphs)] = fill(GNNGraph([[1,2]]), length((ii-1) * batchsize + 1 : min(ii * batchsize, ngraphs)))
    end
    return g
end

"""
    generate_graph(Sample, rawT, predictT, EQ, sample_time)

Creates a single seismic waveform graph for inference. 

# Arguments
- `S::SeisData`: SeisData. 
- `rawT::AbstractFloat`: Time window of raw data to make prediction, seconds
- `predictT::AbstractFloat`: Forward prediction time, seconds
- `EQ::EQLoc`: Location of earthquake
- `sample_time::DateTime`: Last time of sample window 

# Optional 
-`k::Integer`: number of nearest neigherbors per station
-`maxdist::Real`: Connect all stations within maxdist, meters 
-`α::Real`: clustering exponent for long-range connection  
-`q::Integer`: Number of long-range connections per station
-`logpga::Bool`: Make target of graph log10 of peak ground acceleration (PGA)
-`magntiude::Real`: Earthquake magnitude for magnitude-dependent attentuation
-`jma_sort::Bool`: Filter PGA by largest shaking over 0.3 seconds   

# Returns 
- `g::GNNGraph`: Seismic station graphs
- `distance_from_earthquake::Vector`: Stations in each seismic station graph  
- `lon::Vector`: Station longitude 
- `lat::Vector`: Station latitude 
"""
function generate_graph(
    S::SeisData,
    rawT::AbstractFloat,
    predictT::AbstractFloat,
    EQ::EQLoc,
    sample_time::DateTime;
    k::Integer=10,          # K-nearest neighbors  
    maxdist::Real=20000.0,  # meters 
    α::Real=2.0,
    q::Integer=1,
    normalize::Bool=false, 
    logpga::Bool=false,
    magnitude::Real=7.0,
    jma_sort::Bool=false,
)   

    # grab E, N, and Z components 
    E_idx = findall(endswith.(S.id, "E")) 
    N_idx = findall(endswith.(S.id, "N")) 
    Z_idx = findall(endswith.(S.id, "Z")) 

    # distance from earthquake 
    lon = zeros(length(Z_idx))
    lat = zeros(length(Z_idx))
    for (ii,zz) in enumerate(Z_idx)
        lon[ii] = S.loc[zz].lon
        lat[ii] = S.loc[zz].lat
    end
    distance_from_earthquake = distance(lat, lon, EQ) ./ 1000
    distance_idx = sortperm(distance_from_earthquake)
    distance_from_earthquake = distance_from_earthquake[distance_idx]
    lon = lon[distance_idx]
    lat = lat[distance_idx]
    E_idx = E_idx[distance_idx]
    N_idx = N_idx[distance_idx]
    Z_idx = Z_idx[distance_idx]

    # create adjacency matrix 
    station_distance = distance(lat, lon)
    adjK = knn_adjacency(station_distance, k)
    adjD = adjacency(station_distance, maxdist)
    adj = adjK + adjD 
    g = SimpleDiGraph(adj)

    # add shortcuts, if needed 
    if q > 0
        kleinberg_navigation!(g, α, q)
    end
    adj = adjacency(g, kanno_2006_amplitude(station_distance ./ 1000.0, magnitude))

    # get times for extraction 
    start_time = Millisecond(-rawT * 1000)
    end_time = Millisecond(predictT * 1000)
    t = (u2d.(SeisIO.t_expand(S.t[1], S.fs[1]) .* 1e-6) .- sample_time) 
    T_idx = findall(start_time .< t .<= end_time)
    raw_idx = 1:round(Int, rawT * S.fs[1])

    # transform into 4D tensor 
    X = S2T(S, E_idx, N_idx, Z_idx, T_idx) 

    # extract raw + pga data 
    raw = X[raw_idx, :, :, :]

    # convert 3-component acceleration to vector sum 
    ir3 = dropdims(sqrt.(sum(X .^ 2, dims=2)), dims=(2,3))

    # sort PGA that occur for >= 0.3 second 
    if jma_sort 
        third_idx = round(Int, 0.3 * S.fs[1])
        pga = partialsort_n(ir3, third_idx)
    else
        pga = dropdims(maximum(ir3, dims=1), dims=1) 
    end

    if logpga
        pga = log10.(pga)
    end

    if normalize
        raw = maxnorm12(raw)
    end

    g = GNNGraph(adj, ndata = (x=raw,), gdata= (u=pga,))
    return g, distance_from_earthquake, lon, lat 
end

function fast_noise_batch(graphs::AbstractArray, nstations::Integer)
    
    N_graphs = length(graphs)
    T1 = eltype(graphs[1].ndata.x)
    
    # preallocate data matrix
    x_size = collect(size(graphs[1].ndata.x))
    x_size[end] *= N_graphs
    x_size = Tuple(x_size)
    x = zeros(T1, x_size)

    # preallocate graph data matrix 
    T2 = eltype(graphs[1].gdata.u)
    u_size = (size(graphs[1].gdata.u, 1), N_graphs)
    u = zeros(T2, u_size)

    # preallocate graphs 
    sources = Int[]
    targets = Int[]
    graph_indicator = Int[]
    weights = Float64[]

    # fill graphs 
    for ii in 1:N_graphs 
        x[:,:,:,(ii - 1) * nstations + 1 : ii * nstations] .= graphs[ii].ndata.x 
        u[:,ii] .= graphs[ii].gdata.u[:,1]
        append!(sources, graphs[ii].graph[1] .+ (ii - 1) * nstations)
        append!(targets, graphs[ii].graph[2] .+ (ii - 1) * nstations)
        append!(graph_indicator, ones(Int64, nstations) .* ii)
        append!(weights, graphs[ii].graph[3])
    end

    G = GNNGraph((sources, targets, weights), ndata=(x=x,), gdata=(u=u), graph_indicator=graph_indicator)
    return G
end

function read_noise(noise_file::String, noise_samples::Int; maximum_noise_amplitude::Real = 1.0)
    # filter channels by amplitude and length 
    S = read_data(noise_file)
    noise_pga = maximum.(S.x)
    noise_lengths = length.(S.x)
    noise_indices = findall((noise_pga .< maximum_noise_amplitude) .& (noise_lengths .> noise_samples))
    isempty(noise_indices) && return Matrix{Float32}[]

    # ensure that each station has 3 channels remaining 
    S_noise = S[noise_indices]
    channels = S_noise.id 
    stations = [join(split(channel, ".")[1:2], ".") for channel in channels]
    unique_stations = unique(stations)
    station_counts = zeros(Int, length(unique_stations))
    for ii in 1:length(station_counts)
        station_counts[ii] = length(findall(unique_stations[ii] .== stations))
    end
    station_unique_indices = findall(station_counts .== 3)
    if isempty(station_unique_indices)
        return Matrix{Float32}[]
    else
        station_keep_indices = findall(in(unique_stations[station_unique_indices]), stations) 
        S_noise = S_noise[station_keep_indices]
    end
    sort!(S_noise)

    # create array of 3 x N matrices (number samples x ENZ)
    noise_array = Array{Matrix{Float32}}(undef, S_noise.n ÷ 3)
    for ii in 1:S_noise.n ÷ 3
        # ensure that all channels are from same station 
        noise_channels = S_noise.id[(ii - 1) * 3 + 1:ii*3]
        noise_stations = [n[1:end-1] for n in noise_channels]
        if length(Set(noise_stations)) == 1
            noise_array[ii] = hcat(S_noise.x[(ii - 1) * 3 + 1:ii*3]...)
        end
    end
    
    return noise_array 
end

"""

  S2T(A, n)

Convert SeisData `S` to a 4-D tensor.
"""
function S2T(
    S::SeisData, 
    E_idx::AbstractArray, 
    N_idx::AbstractArray, 
    Z_idx::AbstractArray, 
    T_idx::AbstractArray,
)
    X = zeros(Float32, length(T_idx), 3, 1, length(E_idx))
    for ii in eachindex(E_idx)
        E = @views S.x[E_idx[ii]]
        N = @views S.x[N_idx[ii]]
        Z = @views S.x[Z_idx[ii]]
        X[:,1,:,ii] .= E[T_idx]
        X[:,2,:,ii] .= N[T_idx]
        X[:,3,:,ii] .= Z[T_idx]
    end
    return X 
end

"""

  partialsort_n(A, n)

Partially sort the matrix `A` to extract the `n`th largest value in each column.
"""
function partialsort_n(A::Matrix, n::Integer)
    Nrows, Ncols = size(A)
    out = zeros(eltype(A), Ncols)
    for ii in eachindex(out)
        out[ii] = partialsort(A[:,ii], n, rev=true)
    end
    return out
end
