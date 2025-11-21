#!/usr/bin/env julia

using GRAPES
using GRAPES: load_kiknet, read_hypocenter, JMA_travel_time, generate_graph_batch, generate_test_batch, moving_abs_maximum
using Dates
using DataFrames
using Statistics
using ProgressMeter
using SeisIO
using SeisIO.Quake: EQLoc
using DSP
using JLD2
using Logging
using Printf
using Base.Threads

const TARGET_FS = 100.0
const START_TAPER_SECONDS = 1.0
const END_TAPER_SECONDS = 10.0
const MIN_PRE_NOISE_SECONDS = 4.0
const NOISE_WINDOW_SECONDS = 4.0
const MOVING_MAX_WINDOW_SECONDS = 0.5
const RAW_WINDOW_SECONDS = 4.0         # waveform window fed to GRAPES graphs
const PREDICT_WINDOW_SECONDS = 40.0    # PGA prediction horizon
const MINUTE_FORMAT = dateformat"yyyymmddHHMMSS"

struct MinuteJob
    timestamp::DateTime
    knet_path::String
    kik_path::String
end

struct HypoMatch
    row::DataFrameRow
    eq::EQLoc
end

struct CLIOptions
    dry_run::Bool
    workers::Int
end

struct PipelineConfig
    root::String
    knet_root::String
    kik_root::String
    seisio_root::String
    graph_root::String
    hypo_root::String
    hp_filter
end

struct ChannelArtifacts
    pre_noise::Vector{Float32}
    post_noise::Vector{Float32}
    p_arrival::DateTime
end

struct PipelineStats
    processed::Atomic{Int}
    skipped_existing::Atomic{Int}
    failed::Atomic{Int}
    graph_only::Atomic{Int}
    dry_matches::Atomic{Int}
end

PipelineStats() = PipelineStats(
    Atomic{Int}(0),
    Atomic{Int}(0),
    Atomic{Int}(0),
    Atomic{Int}(0),
    Atomic{Int}(0),
)

const catalog_cache = Dict{Int,DataFrame}()
const catalog_lock = ReentrantLock()
const travel_time_lock = ReentrantLock()
const seisio_lock = ReentrantLock()
const graph_lock = ReentrantLock()
const progress_lock = ReentrantLock()

function parse_cli(args::Vector{String})
    dry = any(==("--dry-run"), args)
    workers = Threads.nthreads()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--dry-run"
            i += 1
            continue
        elseif arg == "--workers"
            i == length(args) && error("--workers requires a value")
            workers = parse(Int, args[i + 1])
            i += 2
            continue
        elseif arg == "--help"
            println("Usage: julia --project -t auto preprocess.jl [--dry-run] [--workers N]")
            exit(0)
        else
            error("Unknown argument: $arg")
        end
    end
    workers = max(1, min(workers, Threads.nthreads()))
    return CLIOptions(dry, workers)
end

function minute_stamp(dt::DateTime)
    return Dates.format(dt, MINUTE_FORMAT)
end

function enumerate_paired_minutes(knet_root::String, kik_root::String)
    jobs = MinuteJob[]
    knet_years = filter(x -> isdir(joinpath(knet_root, x)), readdir(knet_root))
    kik_years = filter(x -> isdir(joinpath(kik_root, x)), readdir(kik_root))
    for year in sort!(collect(intersect(knet_years, kik_years)))
        knet_year_dir = joinpath(knet_root, year)
        kik_year_dir = joinpath(kik_root, year)
        knet_months = filter(x -> isdir(joinpath(knet_year_dir, x)), readdir(knet_year_dir))
        kik_months = filter(x -> isdir(joinpath(kik_year_dir, x)), readdir(kik_year_dir))
        for month in sort!(collect(intersect(knet_months, kik_months)))
            knet_month_dir = joinpath(knet_year_dir, month)
            kik_month_dir = joinpath(kik_year_dir, month)
            knet_entries = Dict{String,String}()
            kik_entries = Dict{String,String}()
            for file in sort(filter(f -> endswith(f, ".knt.tar.gz"), readdir(knet_month_dir)))
                stamp = first(split(file, "."))
                try
                    _ = DateTime(stamp, MINUTE_FORMAT)
                    knet_entries[stamp] = joinpath(knet_month_dir, file)
                catch
                end
            end
            for file in sort(filter(f -> endswith(f, ".kik.tar.gz"), readdir(kik_month_dir)))
                stamp = first(split(file, "."))
                try
                    _ = DateTime(stamp, MINUTE_FORMAT)
                    kik_entries[stamp] = joinpath(kik_month_dir, file)
                catch
                end
            end
            for stamp in sort!(collect(intersect(keys(knet_entries), keys(kik_entries))))
                timestamp = DateTime(stamp, MINUTE_FORMAT)
                push!(jobs, MinuteJob(timestamp, knet_entries[stamp], kik_entries[stamp]))
            end
        end
    end
    sort!(jobs, by = job -> job.timestamp)
    return jobs
end

function load_catalog(year::Int, config::PipelineConfig)
    if haskey(catalog_cache, year)
        return catalog_cache[year]
    end
    lock(catalog_lock) do
        return get!(catalog_cache, year) do
            file = joinpath(config.hypo_root, "h$(year)")
            isfile(file) || error("Missing hypocenter catalog $file")
            read_hypocenter(file)
        end
    end
end

function row_to_eqloc(row::DataFrameRow)
    (ismissing(row.LAT) || ismissing(row.LON) || ismissing(row.DEPTH)) && return nothing
    lat = Float64(row.LAT)
    lon = Float64(row.LON)
    dep = Float64(row.DEPTH)
    nsta = ismissing(row.NSTA) ? 0 : Int(row.NSTA)
    return EQLoc(; lat=lat, lon=lon, dep=dep, nst=nsta, datum="JMA", src="JMA", typ="catalog")
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

function match_hypocenter(minute::DateTime, prelim::EQLoc, config::PipelineConfig)
    start_time = minute
    stop_time = minute + Second(60)
    candidate_years = unique((year(start_time), year(stop_time)))
    best_row = nothing
    best_eq = nothing
    best_dist = typemax(Float64)
    best_dt = typemax(Int)
    target_mid = minute + Second(30)
    for yr in candidate_years
        catalog = load_catalog(yr, config)
        mask = (catalog.ORIGIN_DATETIME .>= start_time) .& (catalog.ORIGIN_DATETIME .<= stop_time)
        idxs = findall(mask)
        for idx in idxs
            row = catalog[idx, :]
            mi = row.MAXIMUM_INTENSITY
            if ismissing(mi) || mi < 2
                continue
            end
            mag = row.MAG
            if ismissing(mag) || mag < 3.0
                continue
            end
            nsta = row.NSTA
            if ismissing(nsta) || nsta < 30
                continue
            end
            eq = row_to_eqloc(row)
            eq === nothing && continue
            dist = hypo_distance(eq, prelim)
            time_delta = abs(Dates.value(row.ORIGIN_DATETIME - target_mid))
            if dist < best_dist || (dist ≈ best_dist && time_delta < best_dt)
                best_row = row
                best_eq = eq
                best_dist = dist
                best_dt = time_delta
            end
        end
    end
    return isnothing(best_row) ? nothing : HypoMatch(best_row, best_eq)
end

function minute_output_paths(config::PipelineConfig, minute::DateTime)
    year_str = lpad(string(year(minute)), 4, '0')
    month_str = lpad(string(month(minute)), 2, '0')
    stamp = minute_stamp(minute)
    seisio_dir = joinpath(config.seisio_root, year_str, month_str)
    graph_dir = joinpath(config.graph_root, year_str, month_str)
    return (
        seisio_dir=seisio_dir,
        graph_dir=graph_dir,
        seisio_path=joinpath(seisio_dir, "$stamp.seisio"),
        graph_path=joinpath(graph_dir, "$stamp.graph.jld2"),
        test_path=joinpath(graph_dir, "$stamp.test.jld2"),
    )
end

function load_preliminary_header(job::MinuteJob)
    _, hdr = load_kiknet(job.knet_path)
    return hdr
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
        id_parts = split(S.id[ii], '.')
        if !isempty(id_parts)
            chan = id_parts[end]
            id_parts[end] = component_alias(chan)
            new_id = join(id_parts, ".")
            S.id[ii] = new_id
            S.name[ii] = new_id
        end
    end
    return nothing
end

function component_alias(chan::AbstractString)
    c = uppercase(chan)
    if startswith(c, "E")
        return "E"
    elseif startswith(c, "N")
        return "N"
    elseif startswith(c, "U") || startswith(c, "Z")
        return "Z"
    else
        return chan
    end
end

function deduplicate_channels(S::SeisData)
    seen = Dict{String,Bool}()
    keep = trues(S.n)
    for ii in 1:S.n
        id = S.id[ii]
        if haskey(seen, id)
            keep[ii] = false
        else
            seen[id] = true
        end
    end
    keep_idx = findall(keep)
    if length(keep_idx) == S.n
        return S
    end
    return S[keep_idx]
end

function sanitize_noise_segment(noise::Vector{Float32}, fs::Float64)
    if isempty(noise)
        return noise
    end
    window = max(1, round(Int, MOVING_MAX_WINDOW_SECONDS * fs))
    env = moving_abs_maximum(noise, window)
    med = median(env)
    thresh = (med == 0 ? maximum(env) : med) * 3
    if thresh <= 0
        return copy(noise)
    end
    cleaned = copy(noise)
    fill_value = median(noise)
    for (val, idx) in zip(env, eachindex(env))
        if val > thresh
            cleaned[idx] = fill_value
        end
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

function preprocess_channels!(S::SeisData, match::HypoMatch, job::MinuteJob, config::PipelineConfig)
    if S.n == 0
        return SeisData(), ChannelArtifacts[]
    end
    ungap!(S, m=false, tap=false)
    resample!(S, fs=TARGET_FS)
    normalize_channel_ids!(S)
    S = deduplicate_channels(S)
    p_travel = lock(travel_time_lock) do
        JMA_travel_time(match.eq, S, "p")
    end
    keep = trues(S.n)
    artifacts = Vector{Union{ChannelArtifacts,Nothing}}(undef, S.n)
    for ii in 1:S.n
        fs = S.fs[ii]
        if fs <= 0
            keep[ii] = false
            artifacts[ii] = nothing
            @warn "Dropping $(S.id[ii]) for $(minute_stamp(job.timestamp)): invalid sample rate"
            continue
        end
        start_us = SeisIO.starttime(S.t[ii], fs)
        start_dt = SeisIO.u2d(start_us * 1e-6)
        travel_time = p_travel[ii]
        p_arrival = match.row.ORIGIN_DATETIME + Millisecond(round(Int, travel_time * 1000))
        pre_noise_seconds = (p_arrival - start_dt).value / 1000.0
        if pre_noise_seconds <= 0
            keep[ii] = false
            artifacts[ii] = nothing
            @warn "Dropping $(S.id[ii]) for $(minute_stamp(job.timestamp)): P arrival precedes data"
            continue
        end
        samples = length(S.x[ii])
        pre_samples = min(samples, floor(Int, pre_noise_seconds * fs))
        if pre_samples <= 0
            keep[ii] = false
            artifacts[ii] = nothing
            @warn "Dropping $(S.id[ii]) for $(minute_stamp(job.timestamp)): missing pre-P noise"
            continue
        end
        noise_mean = mean(@view S.x[ii][1:pre_samples])
        S.x[ii] .-= noise_mean
        start_trim, end_trim = apply_asymmetric_taper!(S.x[ii], fs)
        filtered = Float32.(filt(config.hp_filter, Float64.(S.x[ii])))
        S.x[ii] = filtered
        if !trim_tapered_segments!(S, ii, start_trim, end_trim)
            keep[ii] = false
            artifacts[ii] = nothing
            @warn "Dropping $(S.id[ii]) for $(minute_stamp(job.timestamp)): waveform too short after taper trim"
            continue
        end
        new_start_dt = SeisIO.u2d(SeisIO.starttime(S.t[ii], fs) * 1e-6)
        new_pre_seconds = (p_arrival - new_start_dt).value / 1000.0
        if new_pre_seconds < MIN_PRE_NOISE_SECONDS
            keep[ii] = false
            artifacts[ii] = nothing
            @warn "Dropping $(S.id[ii]) for $(minute_stamp(job.timestamp)): insufficient pre-P noise ($(round(new_pre_seconds, digits=2)) s)"
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
        artifacts[ii] = ChannelArtifacts(pre_segment, post_segment, p_arrival)
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

function component_symbol(id::String)
    parts = split(id, '.')
    chan = isempty(parts) ? id : parts[end]
    c = uppercase(chan)
    if startswith(c, "E")
        return 'E'
    elseif startswith(c, "N")
        return 'N'
    elseif startswith(c, "U") || startswith(c, "Z")
        return 'Z'
    end
    return nothing
end

function enforce_complete_triplets!(S::SeisData, artifacts::Vector{ChannelArtifacts}, job::MinuteJob)
    groups = Dict{String, Vector{Int}}()
    for (idx, id) in enumerate(S.id)
        base = station_base(id)
        push!(get!(groups, base, Int[]), idx)
    end
    keep = Int[]
    keep_artifacts = ChannelArtifacts[]
    for (_, idxs) in groups
        comps = Dict('E'=>Int[], 'N'=>Int[], 'Z'=>Int[])
        for idx in idxs
            comp = component_symbol(S.id[idx])
            if isnothing(comp)
                @warn "Dropping $(S.id[idx]) for $(minute_stamp(job.timestamp)): unknown component code"
                continue
            end
            push!(comps[comp], idx)
        end
        if all(!isempty, values(comps))
            append!(keep, vcat(comps['E'][1:1], comps['N'][1:1], comps['Z'][1:1]))
            push!(keep_artifacts, artifacts[comps['E'][1]])
            push!(keep_artifacts, artifacts[comps['N'][1]])
            push!(keep_artifacts, artifacts[comps['Z'][1]])
        else
            for idx in idxs
                comp = component_symbol(S.id[idx])
                if isnothing(comp) || isempty(comps[comp])
                    @warn "Dropping $(S.id[idx]) for $(minute_stamp(job.timestamp)): incomplete station components"
                end
            end
        end
    end
    if isempty(keep)
        return SeisData(), ChannelArtifacts[]
    end
    order = sortperm(keep)
    keep = keep[order]
    keep_artifacts = keep_artifacts[order]
    return S[keep], keep_artifacts
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

function remove_gain_and_convert!(S::SeisData)
    for ii in 1:S.n
        gain = S.gain[ii]
        if gain != 0 && gain != 1
            S.x[ii] ./= gain
        end
        units = lowercase(String(S.units[ii]))
        if units == "m/s^2" || units == "m/s^2 "
            S.x[ii] .*= 100
            S.units[ii] = "cm/s^2"
        elseif units in ("gal", "cm/s^2")
            S.units[ii] = "cm/s^2"
        end
        S.gain[ii] = 1.0
    end
    return nothing
end

function write_seisio(path::String, S::SeisData)
    mkpath(dirname(path))
    lock(seisio_lock) do
        wseis(path, S)
    end
end

function write_graph(graph_path::String, seisio_path::String, match::HypoMatch)
    mkpath(dirname(graph_path))
    lock(graph_lock) do
        graphs, stations, starttimes = generate_graph_batch(
            seisio_path,
            RAW_WINDOW_SECONDS,
            PREDICT_WINDOW_SECONDS,
            match.eq,
            match.row.ORIGIN_DATETIME,
            match.row.MAXIMUM_INTENSITY;
            k=20,
            maxdist=30000.0,
            batchsize=16,
            logpga=true,
            magnitude=match.row.MAG,
        )
        if graphs === nothing
            return false
        end
        JLD2.jldsave(graph_path; graphs=graphs, stations=stations, starttimes=starttimes)
        return true
    end
end

function write_test_graph(test_path::String, seisio_path::String, match::HypoMatch)
    mkpath(dirname(test_path))
    lock(graph_lock) do
        all_graphs, window_start_times, window_end_times, distance_from_earthquake, lon, lat = generate_test_batch(
            seisio_path,
            RAW_WINDOW_SECONDS,
            PREDICT_WINDOW_SECONDS,
            match.eq,
            match.row.ORIGIN_DATETIME;
            k=20,
            maxdist=30000.0,
            logpga=true,
            magnitude=match.row.MAG,
        )
        if all_graphs === nothing
            return false
        end
        JLD2.jldsave(test_path; all_graphs=all_graphs, window_start_times=window_start_times, window_end_times=window_end_times, distance_from_earthquake=distance_from_earthquake, lon=lon, lat=lat)
        return true
    end
end

function dry_run_minute(job::MinuteJob, config::PipelineConfig, stats::PipelineStats)
    hdr = load_preliminary_header(job)
    match = match_hypocenter(job.timestamp, hdr.loc, config)
    if match === nothing
        atomic_add!(stats.failed, 1)
        @warn "No hypocenter match for $(minute_stamp(job.timestamp))"
    else
        atomic_add!(stats.dry_matches, 1)
        row = match.row
        summary = @sprintf(
            "%s matched to origin=%s lat=%.3f lon=%.3f depth=%.1fkm M=%.2f I=%.1f",
            minute_stamp(job.timestamp),
            string(row.ORIGIN_DATETIME),
            Float64(row.LAT),
            Float64(row.LON),
            Float64(row.DEPTH),
            Float64(row.MAG),
            Float64(row.MAXIMUM_INTENSITY),
        )
        println(summary)
    end
end

function process_minute(job::MinuteJob, config::PipelineConfig, stats::PipelineStats)
    paths = minute_output_paths(config, job.timestamp)
    if isfile(paths.seisio_path) && isfile(paths.graph_path) && isfile(paths.test_path)
        atomic_add!(stats.skipped_existing, 1)
        @info "Skipping $(minute_stamp(job.timestamp)): outputs exist"
        return
    end
    S, hdr = load_waveforms(job)
    match = match_hypocenter(job.timestamp, hdr.loc, config)
    if match === nothing
        atomic_add!(stats.failed, 1)
        @warn "No hypocenter match for $(minute_stamp(job.timestamp))"
        return
    end
    S_filtered, artifacts = preprocess_channels!(S, match, job, config)
    if S_filtered.n == 0
        atomic_add!(stats.failed, 1)
        @warn "Preprocessing removed all channels for $(minute_stamp(job.timestamp))"
        return
    end
    S_clean, artifacts = enforce_complete_triplets!(S_filtered, artifacts, job)
    if S_clean.n == 0
        atomic_add!(stats.failed, 1)
        @warn "Incomplete stations after QC for $(minute_stamp(job.timestamp))"
        return
    end
    align_time_windows!(S_clean, artifacts, job)
    remove_gain_and_convert!(S_clean)
    write_seisio(paths.seisio_path, S_clean)
    graph_ok = write_graph(paths.graph_path, paths.seisio_path, match)
    test_ok = write_test_graph(paths.test_path, paths.seisio_path, match)
    if graph_ok && test_ok
        atomic_add!(stats.processed, 1)
        @info "Processed $(minute_stamp(job.timestamp)) -> $(paths.seisio_path)"
    else
        atomic_add!(stats.graph_only, 1)
        @warn "Graph generation skipped for $(minute_stamp(job.timestamp))"
    end
end

function worker_loop!(jobs::Channel{MinuteJob}, config::PipelineConfig, opts::CLIOptions, stats::PipelineStats, progress::Progress)
    for job in jobs
        try
            if opts.dry_run
                dry_run_minute(job, config, stats)
            else
                process_minute(job, config, stats)
            end
        catch err
            atomic_add!(stats.failed, 1)
            @error "Failed $(minute_stamp(job.timestamp))" exception=(err, catch_backtrace())
        end
        lock(progress_lock) do
            ProgressMeter.next!(progress)
        end
    end
end

function run_pipeline(opts::CLIOptions, config::PipelineConfig)
    jobs = enumerate_paired_minutes(config.knet_root, config.kik_root)
    isempty(jobs) && (println("No paired minutes found."); return)
    mode = opts.dry_run ? "Dry-run" : "Processing"
    progress = Progress(length(jobs); desc="$mode GRAPES minutes", dt=0.5)
    stats = PipelineStats()
    job_channel = Channel{MinuteJob}(length(jobs))
    @async begin
        for job in jobs
            put!(job_channel, job)
        end
        close(job_channel)
    end
    @sync begin
        for _ in 1:opts.workers
            @spawn worker_loop!(job_channel, config, opts, stats, progress)
        end
    end
    ProgressMeter.finish!(progress)
    println("processed=$(stats.processed[]), skipped_existing=$(stats.skipped_existing[]), failed=$(stats.failed[]), graph_skipped=$(stats.graph_only[])")
end

function build_config()
    root = @__DIR__
    hp = digitalfilter(Highpass(0.25, fs=TARGET_FS), Butterworth(4))
    return PipelineConfig(
        root,
        joinpath(root, "data", "knet"),
        joinpath(root, "data", "kik"),
        joinpath(root, "data", "seisio"),
        joinpath(root, "data", "jld2"),
        joinpath(root, "data", "h"),
        hp,
    )
end

function main()
    opts = parse_cli(copy(ARGS))
    config = build_config()
    run_pipeline(opts, config)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
