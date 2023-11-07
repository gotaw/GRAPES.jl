"""
This script applies the following processing to channels for the 2019 M7.1 Ridgecrest earthquake. 
    1. Read waveforms 
    2. Remove data gaps 
    3. Remove pre-earthquake mean from waveform 
    4. Apply 5-second Tukey taper to beginning and end of waveform  
    5. Downsample to 100 Hz 
    6. Sync all waveforms to begin 10 seconds before origin time 
    7. Remove gain from waveform 
    8. Convert to cm/s^2 
    9. Ensure each station has 3 channels (ENZ)
    10. Write all channels to single .seisio file 
"""

println("Processing Ridgecrest waveform data")

# time of ridgecrest earthquake 
origin_time = DateTime(2019, 7, 6, 3, 19, 53)

# load channels for 2019 M7.1 Ridgecrest earthquake
waveforms = readdir(download_cache, join=true)
S = read_data(waveforms)
Scop = deepcopy(S)
merge!(S)
# remove gaps 
ungap!(S, m=false, tap=false)
# causal mean removal 
for ii in 1:S.n 
    nan_idx = findall(isnan.(S.x[ii]))
    S.x[ii][nan_idx] .= 0
    S.x[ii] .-= mean(S.x[ii][1:1000])
end
taper!(S, t_max=5.0)
resample!(S, fs=100.0)
sync!(S, s=origin_time - Second(10))
rescale!(S, 1.0)

# round all traces to nearest sample, then sync 
for ii in 1:S.n
    nearest_datetime = round(u2d(SeisIO.starttime(S.t[ii], S.fs[ii]) * 1e-6), Millisecond(10))
    S.t[ii][1,2] = d2u(nearest_datetime) * 1e6
end
sync!(S, s="last", t="first")

# check for zero amplitude 
max_amp = [maximum(abs.(S.x[ii])) for ii in 1:S.n]
amp_idx = findall(max_amp .!= 0.0 )
S = S[amp_idx]

# convert to cm/s^2 
for ii in 1:S.n 
    S.x[ii] .*= 100
end

# ensure each station has 3 channels 
unique_station = unique([join(split(S.id[ii], ".")[1:2], ".") for ii in 1:S.n])
station_count = Dict()
for sta in unique_station
    station_count[sta] = 0
end

for ii in 1:S.n 
    sta = join(split(S.id[ii], ".")[1:2], ".")
    station_count[sta] += 1 
end
remove_sta = collect(keys(station_count))[findall(values(station_count) .!= 3)]
station_to_keep = [ii for ii in 1:S.n if all( .!contains.(join(split(S.id[ii], ".")[1:2], "."), remove_sta)) ]
S = S[station_to_keep]

# write combined waveforms to file 
if isfile(waveform_path)
    @warn "Over-writing processed data for Ridgecrest EQ"
    wseis(waveform_path, S)
end