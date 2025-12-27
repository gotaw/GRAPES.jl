#!/usr/bin/env julia

using GRAPES
using GRAPES: load_kiknet, read_hypocenter, JMA_travel_time, generate_graph_batch, generate_noise_batch, moving_abs_maximum
using Dates
using DataFrames
using Statistics
using ProgressMeter
using SeisIO
using SeisIO.Quake: EQLoc
using DSP
using JLD2
using Logging

const TARGET_FS = 100.0
const START_TAPER_SECONDS = 1.0
const END_TAPER_SECONDS = 10.0
const MIN_PRE_NOISE_SECONDS = 4.0
const NOISE_WINDOW_SECONDS = 4.0
const MOVING_MAX_WINDOW_SECONDS = 0.5
const RAW_WINDOW_SECONDS = 4.0         # waveform window fed to GRAPES graphs
const PREDICT_WINDOW_SECONDS = 40.0    # PGA prediction horizon
const MINUTE_FORMAT = dateformat"yyyymmddHHMMSS"
const GRAPH_BATCHSIZE = 16
const NOISE_BATCHSIZE = 2
const BASELINE_INTENSITY = 2.0

struct MinuteJob
    timestamp::DateTime
    knet_path::String
    kik_path::String
end

struct HypoMatch
    row::DataFrameRow
    eq::EQLoc
end

struct PipelineConfig
    root::String
    external_root::String
    interim_root::String
    processed_root::String
    hp_filter
end

external_root(config::PipelineConfig, subdir::AbstractString) = joinpath(config.external_root, subdir)

struct MinutePaths
    seis_path::String
    graph_path::String
    noise_path::String
    failed_path::String
end

struct ChannelArtifacts
    pre_noise::Vector{Float32}
    post_noise::Vector{Float32}
end

struct ProcessedEntry
    timestamp::DateTime
    kind::Symbol
    path::String
    graph_equiv::Int
end


mutable struct PipelineStats
    skipped_existing::Int
    graph_written::Int
    noise_written::Int
    failed_written::Int
    warnings::Vector{String}
end

PipelineStats() = PipelineStats(0, 0, 0, 0, String[])

function record_warn!(stats::PipelineStats, msg::AbstractString)
    push!(stats.warnings, String(msg))
    @debug msg
end

const catalog_cache = Dict{Int,DataFrame}()

function p_arrival_times(S::SeisData, match::HypoMatch)
    travel = JMA_travel_time(match.eq, S, "p")
    origin = match.row.ORIGIN_DATETIME
    return [origin + Millisecond(round(Int, tt * 1000)) for tt in travel]
end

minute_stamp(dt::DateTime) = Dates.format(dt, MINUTE_FORMAT)

stamp_parts(dt::DateTime) = (lpad(string(year(dt)), 4, '0'),
                             lpad(string(month(dt)), 2, '0'),
                             minute_stamp(dt))

list_dirs(root::String) = filter(name -> isdir(joinpath(root, name)), readdir(root))

intersect_dirs(root_a::String, root_b::String) = sort!(collect(intersect(list_dirs(root_a), list_dirs(root_b))))

function collect_minute_entries(dir::String, suffix::String)
    entries = Dict{String,String}()
    for file in sort(filter(f -> endswith(f, suffix), readdir(dir)))
        stamp = split(file, '.'; limit=2)[1]
        try
            DateTime(stamp, MINUTE_FORMAT)
            entries[stamp] = joinpath(dir, file)
        catch
        end
    end
    return entries
end

function enumerate_paired_minutes(knet_root::String, kik_root::String)
    jobs = MinuteJob[]
    for year in intersect_dirs(knet_root, kik_root),
        month in intersect_dirs(joinpath(knet_root, year), joinpath(kik_root, year))
        knet_entries = collect_minute_entries(joinpath(knet_root, year, month), ".knt.tar.gz")
        kik_entries = collect_minute_entries(joinpath(kik_root, year, month), ".kik.tar.gz")
        for stamp in sort!(collect(intersect(keys(knet_entries), keys(kik_entries))))
            push!(jobs, MinuteJob(DateTime(stamp, MINUTE_FORMAT), knet_entries[stamp], kik_entries[stamp]))
        end
    end
    sort!(jobs, by = job -> job.timestamp)
    return jobs
end

function load_catalog(year::Int, config::PipelineConfig)
    if haskey(catalog_cache, year)
        return catalog_cache[year]
    end
    return get!(catalog_cache, year) do
        file = joinpath(external_root(config, "h"), "h$(year)")
        isfile(file) || error("Missing hypocenter catalog $file")
        read_hypocenter(file)
    end
end

function row_to_eqloc(row::DataFrameRow)
    any(ismissing, (row.LAT, row.LON, row.DEPTH)) && return nothing
    nsta = ismissing(row.NSTA) ? 0 : Int(row.NSTA)
    return EQLoc(;
        lat=Float64(row.LAT),
        lon=Float64(row.LON),
        dep=Float64(row.DEPTH),
        nst=nsta,
        datum="JMA",
        src="JMA",
        typ="catalog",
    )
end

function hypo_distance(a::EQLoc, b::EQLoc)
    return GRAPES.Geodesics.surface_distance(
        a.lon,
        a.lat,
        b.lon,
        b.lat,
        GRAPES.Geodesics.EARTH_R_MAJOR_WGS84,
    )
end

function passes_thresholds(row::DataFrameRow)
    mi = row.MAXIMUM_INTENSITY
    mag = row.MAG
    nsta = row.NSTA
    return !ismissing(mi) && mi >= 2 && !ismissing(mag) && mag >= 3.0 && !ismissing(nsta) && nsta >= 30
end

update_best(best, row, eq, dist, dt) =
    (dist < best.dist || (dist ≈ best.dist && dt < best.dt)) ? (row=row, eq=eq, dist=dist, dt=dt) : best

function match_hypocenter(minute::DateTime, prelim::EQLoc, config::PipelineConfig; apply_filters::Bool=true)
    start_time = minute
    stop_time = minute + Second(60)
    candidate_years = unique((year(start_time), year(stop_time)))
    best_any = (row=nothing, eq=nothing, dist=typemax(Float64), dt=typemax(Int))
    best_pass = (row=nothing, eq=nothing, dist=typemax(Float64), dt=typemax(Int))
    target_mid = minute + Second(30)
    for yr in candidate_years
        catalog = load_catalog(yr, config)
        mask = (catalog.ORIGIN_DATETIME .>= start_time) .& (catalog.ORIGIN_DATETIME .<= stop_time)
        for idx in findall(mask)
            row = catalog[idx, :]
            (ismissing(row.MAXIMUM_INTENSITY) || ismissing(row.MAG)) && continue
            eq = row_to_eqloc(row)
            eq === nothing && continue
            dist = hypo_distance(eq, prelim)
            time_delta = abs(Dates.value(row.ORIGIN_DATETIME - target_mid))
            best_any = update_best(best_any, row, eq, dist, time_delta)
            passes_thresholds(row) && (best_pass = update_best(best_pass, row, eq, dist, time_delta))
        end
    end
    if apply_filters
        return best_pass.row === nothing ? (nothing, false) : (HypoMatch(best_pass.row, best_pass.eq), true)
    end
    if best_pass.row !== nothing
        return HypoMatch(best_pass.row, best_pass.eq), true
    end
    return best_any.row === nothing ? (nothing, false) : (HypoMatch(best_any.row, best_any.eq), passes_thresholds(best_any.row))
end

function minute_output_paths(config::PipelineConfig, minute::DateTime)
    year_str, month_str, stamp = stamp_parts(minute)
    base_dir = joinpath(config.interim_root, year_str, month_str)
    return MinutePaths(
        joinpath(base_dir, "$stamp.seisio"),
        joinpath(base_dir, "$stamp.graph.jld2"),
        joinpath(base_dir, "$stamp.noise.jld2"),
        joinpath(base_dir, "$stamp.failed"),
    )
end

function should_skip_interim(paths::MinutePaths)
    return isfile(paths.graph_path) || isfile(paths.noise_path) || isfile(paths.failed_path)
end

function processed_output_path(config::PipelineConfig, split::Symbol, timestamp::DateTime, kind::Symbol)
    year_str, month_str, stamp = stamp_parts(timestamp)
    base_dir = joinpath(config.processed_root, String(split), year_str, month_str)
    suffix = kind == :graph ? ".graph.jld2" : ".noise.jld2"
    return joinpath(base_dir, "$stamp$suffix")
end

function parse_stamp_from_path(path::String)
    stamp = split(basename(path), '.'; limit=2)[1]
    try
        return DateTime(stamp, MINUTE_FORMAT)
    catch
        return nothing
    end
end

function read_graph_equiv(path::String, batchsize::Int)
    isfile(path) || return nothing
    try
        return length(JLD2.load(path, "graphs")) * batchsize
    catch
        return nothing
    end
end

function load_waveforms(job::MinuteJob)
    knet_data, knet_hdr = load_kiknet(job.knet_path)
    kik_data, _ = load_kiknet(job.kik_path)
    S = SeisData(knet_data, kik_data)
    merge!(S)
    return S, knet_hdr
end

function normalize_channel_ids!(S::SeisData)
    for ii in 1:S.n
        parts = split(S.id[ii], '.')
        isempty(parts) && continue
        parts[end] = component_alias(parts[end])
        new_id = join(parts, ".")
        S.id[ii] = new_id
        S.name[ii] = new_id
    end
    return nothing
end

function component_code(chan::AbstractString)
    c = uppercase(chan)
    startswith(c, "E") && return 'E'
    startswith(c, "N") && return 'N'
    (startswith(c, "U") || startswith(c, "Z")) && return 'Z'
    return nothing
end

component_alias(chan::AbstractString) = (code = component_code(chan); code === nothing ? chan : string(code))

function deduplicate_channels(S::SeisData)
    seen = Set{String}()
    keep = [!(id in seen) && (push!(seen, id); true) for id in S.id]
    all(keep) && return S
    return S[findall(keep)]
end

function sanitize_noise_segment(noise::Vector{Float32}, fs::Float64)
    isempty(noise) && return noise
    window = max(1, round(Int, MOVING_MAX_WINDOW_SECONDS * fs))
    env = moving_abs_maximum(noise, window)
    med = median(env)
    thresh = (med == 0 ? maximum(env) : med) * 3
    thresh <= 0 && return copy(noise)
    cleaned = copy(noise)
    fill_value = median(noise)
    @inbounds for (idx, val) in enumerate(env)
        val > thresh && (cleaned[idx] = fill_value)
    end
    return cleaned
end

function synthesize_extension(noise::Vector{Float32}, samples::Int)
    samples <= 0 && return Float32[]
    isempty(noise) && error("Cannot synthesize extension without noise samples")
    ext = Vector{Float32}(undef, samples)
    rev_noise = reverse(noise)
    seqs = (rev_noise, noise)
    signs = (-1.0f0, 1.0f0)
    idx = 1
    iter = 1
    while idx <= samples
        segment = seqs[iter % 2 == 1 ? 1 : 2]
        sign = signs[iter % 2 == 1 ? 1 : 2]
        copy_len = min(samples - idx + 1, length(segment))
        @inbounds ext[idx:idx+copy_len-1] .= sign .* segment[1:copy_len]
        idx += copy_len
        iter += 1
    end
    return ext
end

function apply_asymmetric_taper!(x::Vector{Float32}, fs::Float64)
    start_len = min(length(x) ÷ 2, round(Int, START_TAPER_SECONDS * fs))
    end_len = min(length(x) - start_len - 1, round(Int, END_TAPER_SECONDS * fs))
    if start_len > 0
        ramp = DSP.Windows.tukey(2 * start_len, 0.5)[1:start_len]
        @inbounds @simd for ii in 1:start_len
            x[ii] *= ramp[ii]
        end
    end
    if end_len > 0
        ramp = DSP.Windows.tukey(2 * end_len, 0.5)[end_len+1:end]
        @inbounds @simd for jj in 1:end_len
            idx = length(x) - end_len + jj
            x[idx] *= ramp[jj]
        end
    end
    return start_len, end_len
end

function trim_tapered_segments!(S::SeisData, idx::Int, start_trim::Int, end_trim::Int)
    ch_len = length(S.x[idx])
    if ch_len <= start_trim + end_trim
        return false
    end
    new_data = @view S.x[idx][start_trim+1:ch_len-end_trim]
    S.x[idx] = copy(new_data)
    shift = round(Int, start_trim / S.fs[idx] * 1e6)
    S.t[idx] = Array{Int64,2}([[1 S.t[idx][1,2] + shift]; [length(S.x[idx]) 0]])
    return true
end

function drop_channel!(keep::BitVector, artifacts::Vector{Union{ChannelArtifacts,Nothing}}, idx::Int, job::MinuteJob, id::String, reason::AbstractString)
    keep[idx] = false
    artifacts[idx] = nothing
    @debug "Dropping $id for $(minute_stamp(job.timestamp)): $reason"
end

function preprocess_channels!(S::SeisData, match::HypoMatch, job::MinuteJob, config::PipelineConfig)
    if S.n == 0
        return SeisData(), ChannelArtifacts[]
    end
    ungap!(S, m=false, tap=false)
    resample!(S, fs=TARGET_FS)
    normalize_channel_ids!(S)
    S = deduplicate_channels(S)
    p_arrivals = p_arrival_times(S, match)
    keep = trues(S.n)
    artifacts = Vector{Union{ChannelArtifacts,Nothing}}(undef, S.n)
    for ii in 1:S.n
        fs = S.fs[ii]
        id = S.id[ii]
        if fs <= 0
            drop_channel!(keep, artifacts, ii, job, id, "invalid sample rate")
            continue
        end
        start_us = SeisIO.starttime(S.t[ii], fs)
        start_dt = SeisIO.u2d(start_us * 1e-6)
        p_arrival = p_arrivals[ii]
        pre_noise_seconds = (p_arrival - start_dt).value / 1000.0
        if pre_noise_seconds <= 0
            drop_channel!(keep, artifacts, ii, job, id, "P arrival precedes data")
            continue
        end
        samples = length(S.x[ii])
        pre_samples = min(samples, floor(Int, pre_noise_seconds * fs))
        if pre_samples <= 0
            drop_channel!(keep, artifacts, ii, job, id, "missing pre-P noise")
            continue
        end
        noise_mean = mean(@view S.x[ii][1:pre_samples])
        S.x[ii] .-= noise_mean
        start_trim, end_trim = apply_asymmetric_taper!(S.x[ii], fs)
        filtered = Float32.(filt(config.hp_filter, Float64.(S.x[ii])))
        S.x[ii] = filtered
        if !trim_tapered_segments!(S, ii, start_trim, end_trim)
            drop_channel!(keep, artifacts, ii, job, id, "waveform too short after taper trim")
            continue
        end
        new_start_dt = SeisIO.u2d(SeisIO.starttime(S.t[ii], fs) * 1e-6)
        new_pre_seconds = (p_arrival - new_start_dt).value / 1000.0
        if new_pre_seconds < MIN_PRE_NOISE_SECONDS
            drop_channel!(keep, artifacts, ii, job, id, "insufficient pre-P noise ($(round(new_pre_seconds, digits=2)) s)")
            continue
        end
        p_samples = clamp(floor(Int, new_pre_seconds * fs), 1, length(S.x[ii]))
        noise_len = min(p_samples, round(Int, NOISE_WINDOW_SECONDS * fs))
        noise_start = max(1, p_samples - noise_len + 1)
        pre_segment = copy(@view S.x[ii][noise_start:p_samples])
        pre_segment = sanitize_noise_segment(pre_segment, fs)
        tail_len = min(length(S.x[ii]), round(Int, NOISE_WINDOW_SECONDS * fs))
        post_segment = copy(@view S.x[ii][end-tail_len+1:end])
        post_segment = sanitize_noise_segment(post_segment, fs)
        artifacts[ii] = ChannelArtifacts(pre_segment, post_segment)
    end
    keep_idx = findall(keep .& .!(artifacts .=== nothing))
    if isempty(keep_idx)
        return SeisData(), ChannelArtifacts[]
    end
    filtered = S[keep_idx]
    filtered_artifacts = ChannelArtifacts[artifacts[i]::ChannelArtifacts for i in keep_idx]
    return filtered, filtered_artifacts
end

function station_base(id::String)
    parts = split(id, '.')
    if length(parts) >= 2
        return string(parts[1], ".", parts[2])
    end
    return id
end

component_symbol(id::String) = (parts = split(id, '.'); component_code(isempty(parts) ? id : parts[end]))

function enforce_complete_triplets!(S::SeisData, artifacts::Vector{ChannelArtifacts}, job::MinuteJob)
    groups = Dict{String, Vector{Int}}()
    for (idx, id) in enumerate(S.id)
        base = station_base(id)
        push!(get!(groups, base, Int[]), idx)
    end
    keep = Int[]
    keep_artifacts = ChannelArtifacts[]
    for idxs in values(groups)
        comps = Dict('E'=>Int[], 'N'=>Int[], 'Z'=>Int[])
        for idx in idxs
            comp = component_symbol(S.id[idx])
            if isnothing(comp)
                @debug "Dropping $(S.id[idx]) for $(minute_stamp(job.timestamp)): unknown component code"
                continue
            end
            push!(comps[comp], idx)
        end
        if all(!isempty, values(comps))
            for comp in ('E', 'N', 'Z')
                idx = comps[comp][1]
                push!(keep, idx)
                push!(keep_artifacts, artifacts[idx])
            end
        else
            for idx in idxs
                comp = component_symbol(S.id[idx])
                if isnothing(comp) || isempty(comps[comp])
                    @debug "Dropping $(S.id[idx]) for $(minute_stamp(job.timestamp)): incomplete station components"
                end
            end
        end
    end
    isempty(keep) && return SeisData(), ChannelArtifacts[]
    order = sortperm(keep)
    return S[keep[order]], keep_artifacts[order]
end

function align_time_windows!(S::SeisData, artifacts::Vector{ChannelArtifacts}, job::MinuteJob)
    fs = TARGET_FS
    start_times = [SeisIO.starttime(S.t[ii], S.fs[ii]) for ii in 1:S.n]
    end_times = [SeisIO.endtime(S.t[ii], S.fs[ii]) for ii in 1:S.n]
    global_start = minimum(start_times)
    global_end = maximum(end_times)
    target_samples = round(Int, (global_end - global_start) / 1e6 * fs) + 1
    for ii in 1:S.n
        artifact = artifacts[ii]
        start_us = start_times[ii]
        n_pre = max(0, round(Int, (start_us - global_start) / 1e6 * fs))
        n_current = length(S.x[ii])
        n_post = target_samples - (n_pre + n_current)
        if n_pre > 0
            ext = synthesize_extension(artifact.pre_noise, n_pre)
            S.x[ii] = vcat(ext, S.x[ii])
        end
        if n_post > 0
            ext = synthesize_extension(artifact.post_noise, n_post)
            append!(S.x[ii], ext)
        elseif n_post < 0
            resize!(S.x[ii], target_samples - n_pre)
        end
        if length(S.x[ii]) != target_samples
            resize!(S.x[ii], target_samples)
        end
        # Prefer SI over paper for start/end noise extension alignment
        S.t[ii] = Array{Int64,2}([[1 global_start]; [length(S.x[ii]) 0]])
    end
    return nothing
end

function normalize_units!(segment::Vector{Float32}, gain, units)
    gain != 0 && gain != 1 && (segment ./= gain)
    u = lowercase(String(units))
    if u == "m/s^2" || u == "m/s^2 "
        segment .*= 100
        return "cm/s^2"
    elseif u in ("gal", "cm/s^2")
        return "cm/s^2"
    end
    return String(units)
end

function collect_noise_samples(S::SeisData, match::HypoMatch)
    isempty(S) && return Matrix{Float32}[], Float64[], Float64[], String[]
    p_arrivals = p_arrival_times(S, match)
    noise_array = Matrix{Float32}[]
    lon = Float64[]
    lat = Float64[]
    stations = String[]
    station_groups = Dict{String, Vector{Int}}()
    for (idx, id) in enumerate(S.id)
        push!(get!(station_groups, station_base(id), Int[]), idx)
    end
    for station in sort!(collect(keys(station_groups)))
        idxs = station_groups[station]
        comps = Dict{Char,Int}()
        for idx in idxs
            comp = component_symbol(S.id[idx])
            isnothing(comp) && continue
            comps[comp] = idx
        end
        if !all(c -> haskey(comps, c), ('E', 'N', 'Z'))
            continue
        end
        ref_idx = comps['Z']
        arrival_time = p_arrivals[ref_idx]
        fs = S.fs[ref_idx]
        raw_samples = round(Int, RAW_WINDOW_SECONDS * fs)
        gap_samples = round(Int, MIN_PRE_NOISE_SECONDS * fs)
        start_dt = SeisIO.u2d(SeisIO.starttime(S.t[ref_idx], fs) * 1e-6)
        samples_to_arrival = floor(Int, ((arrival_time - start_dt).value / 1000) * fs)
        noise_end = samples_to_arrival - gap_samples
        if noise_end <= raw_samples
            continue
        end
        start_idx = noise_end - raw_samples + 1
        end_idx = noise_end
        if any(end_idx > length(S.x[i]) || start_idx < 1 || start_idx > length(S.x[i]) for i in values(comps))
            continue
        end
        noise = map(('E', 'N', 'Z')) do comp
            idx = comps[comp]
            segment = sanitize_noise_segment(copy(@view S.x[idx][start_idx:end_idx]), fs)
            normalize_units!(segment, S.gain[idx], S.units[idx])
            segment
        end
        push!(noise_array, hcat(noise...))
        push!(lon, S.loc[ref_idx].lon)
        push!(lat, S.loc[ref_idx].lat)
        push!(stations, station)
    end
    return noise_array, lon, lat, stations
end

function normalize_units!(S::SeisData)
    for ii in 1:S.n
        S.units[ii] = normalize_units!(S.x[ii], S.gain[ii], S.units[ii])
        S.gain[ii] = 1.0
    end
    return nothing
end

function write_graph(graph_path::String, seis_path::String, match::HypoMatch, intensity::Real)
    mkpath(dirname(graph_path))
    graphs, stations, starttimes = generate_graph_batch(
        seis_path,
        RAW_WINDOW_SECONDS,
        PREDICT_WINDOW_SECONDS,
        match.eq,
        match.row.ORIGIN_DATETIME,
        intensity;
        k=20,
        maxdist=30000.0,
        batchsize=GRAPH_BATCHSIZE,
        logpga=true,
        seqpga=true,
        magnitude=match.row.MAG,
    )
    if graphs === nothing
        return false, 0, 0
    end
    JLD2.jldsave(graph_path; graphs=graphs, stations=stations, starttimes=starttimes)
    return true, length(stations), length(graphs) * GRAPH_BATCHSIZE
end

function write_noise_graph(
    noise_path::String,
    noise_array::Vector{Matrix{Float32}},
    lon::Vector{Float64},
    lat::Vector{Float64},
    match::HypoMatch,
    nbatches::Int,
)
    if isempty(noise_array)
        return false
    end
    nstations = min(length(noise_array), 100)
    if nstations < 2
        return false
    end
    noise_samples = size(noise_array[1], 1)
    inputsize = (noise_samples, 3, 1, nstations)
    k = min(20, nstations - 1)
    q = min(1, nstations - 1)
    maxdist = 30000.0
    noise_graphs = generate_noise_batch(
        noise_array,
        lon,
        lat,
        inputsize,
        nbatches;
        nstations=nstations,
        k=k,
        maxdist=maxdist,
        q=q,
        batchsize=NOISE_BATCHSIZE,
        predictT=PREDICT_WINDOW_SECONDS,
        logpga=true,
        seqpga=true,
        magnitude=match.row.MAG,
    )
    mkpath(dirname(noise_path))
    JLD2.jldsave(noise_path; graphs=noise_graphs)
    return true
end

function mark_failed!(paths::MinutePaths, stats::PipelineStats, context::AbstractString)
    for path in (paths.seis_path, paths.graph_path, paths.noise_path)
        if isfile(path)
            try
                rm(path; force=true)
            catch err
                record_warn!(stats, "$context - failed to remove $path: $(err)")
            end
        end
    end
    mkpath(dirname(paths.failed_path))
    if !isfile(paths.failed_path)
        try
            open(paths.failed_path, "w") do _
            end
            stats.failed_written += 1
        catch err
            record_warn!(stats, "$context - failed to write failed marker: $(err)")
        end
    end
end

function run_interim(config::PipelineConfig)
    jobs = enumerate_paired_minutes(external_root(config, "knet"), external_root(config, "kik"))
    stats = PipelineStats()
    progress = Progress(length(jobs); desc="interim", dt=0.5)
    for job in jobs
        stamp = minute_stamp(job.timestamp)
        paths = minute_output_paths(config, job.timestamp)
        try
            if should_skip_interim(paths)
                stats.skipped_existing += 1
                continue
            end
            S, hdr = load_waveforms(job)
            match, passes_filters = match_hypocenter(job.timestamp, hdr.loc, config; apply_filters=false)
            if match === nothing
                record_warn!(stats, "$stamp - no hypocenter")
                mark_failed!(paths, stats, stamp)
                continue
            end
            S_filtered, artifacts = preprocess_channels!(S, match, job, config)
            if S_filtered.n == 0
                record_warn!(stats, "$stamp - all channels removed during preprocessing")
                mark_failed!(paths, stats, stamp)
                continue
            end
            S_clean, artifacts = enforce_complete_triplets!(S_filtered, artifacts, job)
            if S_clean.n == 0
                record_warn!(stats, "$stamp - no complete E/N/Z station triplets")
                mark_failed!(paths, stats, stamp)
                continue
            end
            if passes_filters
                align_time_windows!(S_clean, artifacts, job)
                normalize_units!(S_clean)
                mkpath(dirname(paths.seis_path))
                wseis(paths.seis_path, S_clean)
                graph_ok, _, _ = write_graph(paths.graph_path, paths.seis_path, match, BASELINE_INTENSITY)
                if graph_ok
                    stats.graph_written += 1
                else
                    record_warn!(stats, "$stamp - graph generation failed")
                    mark_failed!(paths, stats, stamp)
                end
            else
                noise_array, noise_lon, noise_lat, _ = collect_noise_samples(S_clean, match)
                noise_ok = write_noise_graph(paths.noise_path, noise_array, noise_lon, noise_lat, match, 1)
                if noise_ok
                    stats.noise_written += 1
                else
                    record_warn!(stats, "$stamp - noise generation failed")
                    mark_failed!(paths, stats, stamp)
                end
            end
        catch err
            record_warn!(stats, "$stamp - unexpected error: $(err)")
            mark_failed!(paths, stats, stamp)
        finally
            ProgressMeter.next!(progress)
        end
    end
    ProgressMeter.finish!(progress)
end

function collect_processed_entries(config::PipelineConfig, stats::PipelineStats)
    entries = ProcessedEntry[]
    if !isdir(config.interim_root)
        record_warn!(stats, "Missing interim root: $(config.interim_root)")
        return entries
    end
    for (root, _, files) in walkdir(config.interim_root), file in files
        kind = endswith(file, ".graph.jld2") ? :graph :
               endswith(file, ".noise.jld2") ? :noise :
               endswith(file, ".failed") ? :failed : nothing
        kind === nothing && continue
        path = joinpath(root, file)
        ts = parse_stamp_from_path(path)
        if ts === nothing
            record_warn!(stats, "Skipping $(String(kind)) with unparseable stamp: $path")
            continue
        end
        graph_equiv = kind == :graph ? read_graph_equiv(path, GRAPH_BATCHSIZE) :
                      kind == :noise ? read_graph_equiv(path, NOISE_BATCHSIZE) : 0
        if kind != :failed && graph_equiv === nothing
            record_warn!(stats, "Missing graphs in $(String(kind)): $path")
        end
        push!(entries, ProcessedEntry(ts, kind, path, graph_equiv === nothing ? 0 : graph_equiv))
    end
    sort!(entries, by = entry -> entry.timestamp)
    return entries
end

function split_by_equiv(entries::Vector{ProcessedEntry})
    total_equiv = sum((entry.graph_equiv for entry in entries); init=0)
    if isempty(entries) || total_equiv == 0
        return 0, 0, total_equiv
    end
    train_cutoff = total_equiv * 0.7
    val_cutoff = total_equiv * 0.9
    cumulative = 0
    train_end = 0
    val_end = 0
    for (idx, entry) in enumerate(entries)
        cumulative += entry.graph_equiv
        if train_end == 0 && cumulative >= train_cutoff
            train_end = idx
        end
        if val_end == 0 && cumulative >= val_cutoff
            val_end = idx
            break
        end
    end
    train_end == 0 && (train_end = length(entries))
    val_end == 0 && (val_end = length(entries))
    return train_end, val_end, total_equiv
end

function slice_entries(entries::Vector{ProcessedEntry}, train_end::Int, val_end::Int)
    train_entries = train_end > 0 ? entries[1:train_end] : ProcessedEntry[]
    val_entries = val_end > train_end ? entries[train_end+1:val_end] : ProcessedEntry[]
    test_entries = val_end < length(entries) ? entries[val_end+1:end] : ProcessedEntry[]
    return train_entries, val_entries, test_entries
end

function equiv_counts(entries::Vector{ProcessedEntry})
    graph_equiv = sum((entry.graph_equiv for entry in entries if entry.kind == :graph); init=0)
    noise_equiv = sum((entry.graph_equiv for entry in entries if entry.kind == :noise); init=0)
    return graph_equiv, noise_equiv
end

function copy_processed_entry(entry::ProcessedEntry, split::Symbol, config::PipelineConfig)
    dest = processed_output_path(config, split, entry.timestamp, entry.kind)
    if isfile(dest)
        return false
    end
    if !isfile(entry.path)
        @debug "Missing interim source: $(entry.path)"
        return false
    end
    mkpath(dirname(dest))
    cp(entry.path, dest; force=false)
    return true
end

function regenerate_train_graph(entry::ProcessedEntry, config::PipelineConfig)
    dest = processed_output_path(config, :train, entry.timestamp, :graph)
    if isfile(dest)
        return false
    end
    paths = minute_output_paths(config, entry.timestamp)
    if !isfile(paths.seis_path)
        @debug "Missing interim seisio: $(paths.seis_path)"
        return false
    end
    year_str, month_str, stamp = stamp_parts(entry.timestamp)
    knet_path = joinpath(external_root(config, "knet"), year_str, month_str, "$stamp.knt.tar.gz")
    kik_path = joinpath(external_root(config, "kik"), year_str, month_str, "$stamp.kik.tar.gz")
    if !isfile(knet_path) || !isfile(kik_path)
        @debug "Missing external archive for $(minute_stamp(entry.timestamp))"
        return false
    end
    _, hdr = load_kiknet(knet_path)
    match, _ = match_hypocenter(entry.timestamp, hdr.loc, config; apply_filters=true)
    if match === nothing
        @debug "No hypocenter match for $(minute_stamp(entry.timestamp))"
        return false
    end
    graph_ok, _, _ = write_graph(dest, paths.seis_path, match, match.row.MAXIMUM_INTENSITY)
    if !graph_ok
        @debug "Failed to regenerate train graph for $(minute_stamp(entry.timestamp))"
        return false
    end
    return true
end

function run_processed(config::PipelineConfig)
    stats = PipelineStats()
    entries = collect_processed_entries(config, stats)
    entries = [entry for entry in entries if entry.kind != :failed]
    sort!(entries, by = entry -> entry.timestamp)
    graph_files = count(entry -> entry.kind == :graph, entries)
    noise_files = count(entry -> entry.kind == :noise, entries)
    train_end, val_end, total_equiv = split_by_equiv(entries)
    total_graph, total_noise = equiv_counts(entries)
    train_val_boundary = train_end > 0 ? minute_stamp(entries[train_end].timestamp) : "none"
    val_test_boundary = val_end > 0 ? minute_stamp(entries[val_end].timestamp) : "none"
    @info(
        "dataset summary",
        files=(graph=graph_files, noise=noise_files),
        equivalent=(graph=total_graph, noise=total_noise),
        split=(train_val=train_val_boundary, val_test=val_test_boundary),
    )

    if isempty(entries) || total_equiv == 0
        return
    end

    train_entries, val_entries, test_entries = slice_entries(entries, train_end, val_end)
    progress = Progress(length(train_entries) + length(val_entries) + length(test_entries); desc="processed", dt=0.5, output=stderr)
    for (split, entries) in ((:train, train_entries), (:val, val_entries), (:test, test_entries))
        for entry in entries
            if split == :train && entry.kind == :graph
                regenerate_train_graph(entry, config)
            else
                copy_processed_entry(entry, split, config)
            end
            ProgressMeter.next!(progress)
        end
    end
    ProgressMeter.finish!(progress)
end

run_pipeline(config::PipelineConfig) = (run_interim(config); run_processed(config))

function build_config()
    root = normpath(joinpath(@__DIR__, ".."))
    hp = digitalfilter(Highpass(0.25, fs=TARGET_FS), Butterworth(4))
    return PipelineConfig(
        root,
        joinpath(root, "data", "external"),
        joinpath(root, "data", "interim"),
        joinpath(root, "data", "processed"),
        hp,
    )
end

main() = run_pipeline(build_config())

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
