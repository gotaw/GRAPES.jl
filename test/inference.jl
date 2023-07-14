"""
This script makes GRAPES PGA predictions for the 2019 M7.1 Ridgecrest earthquake.
    1. Load trained GRAPES model from disk 
    2. Load Ridgecrest data from disk 
    3. Convert waveform data to seismic waveform graph 
    4. Predict shaking using GRAPES 
"""

# equation 2 from Worden et al., 2012
function PGA_MMI(A)
    threshold = 37.15 # cm/s^2 
    if A <= threshold 
        mmi =  1.78 + 1.55 * log10(A)
    else
        mmi = -1.60 + 3.7 * log10(A)
    end
    return clamp(mmi, 1.0, 10.0)
end

# struct to store GRAPES predictions 
struct GRAPES_pred 
    g::GNNGraph 
    distance_from_earthquake::Vector{Float64}
    lon::Vector{Float64}
    lat::Vector{Float64}
    grapes_mmi::Vector{Float64}
    current_mmi::Vector{Float64}
    maximum_mmi::Vector{Float64}
    grapes_error_mmi::Vector{Float64}
    sample_time::DateTime
    relative_time::Float64

    function GRAPES_pred(
        g::GNNGraph, 
        distance_from_earthquake::Vector{Float64},
        lon::Vector{Float64},
        lat::Vector{Float64},
        grapes_log10::Vector{Float64},
        sample_time::DateTime,
        origin_time::DateTime
    )

        grapes_gal = 10 .^ grapes_log10
        current_pga = vec(GRAPES.getpga(g.ndata.x))
        current_gal = 10 .^ current_pga 
        grapes_mmi = PGA_MMI.(grapes_gal)
        current_mmi = PGA_MMI.(current_gal)
        maximum_mmi = PGA_MMI.(10 .^ vec(g.gdata.u))
        grapes_error_mmi = grapes_mmi - maximum_mmi
        relative_time = (sample_time - origin_time) / Millisecond(1000)

        return new(g, distance_from_earthquake, lon, lat, grapes_mmi, current_mmi, maximum_mmi, grapes_error_mmi, sample_time, relative_time)
    end
end

# load GRAPES model 
model = load_GRAPES_model()

# load Ridgecrest waveforms 
waveform_path = joinpath(processed_cache, "SCSN.seisio") 
S = read_data(waveform_path)
filtfilt!(S, rt="Highpass", fl=0.25)

# source parameters for Ridgecrest 
origin_time = DateTime(2019, 7, 6, 3, 19, 53)
sample_times = origin_time .+ Millisecond.((-0.0:15.0) .* 1000)
relative_times = (sample_times .- origin_time) ./ Millisecond(1000)
ridge_loc = EQLoc(lat=35.77, lon = -117.599, dep=8.0)

# convert to seismic waveform graph 
rawT = 4.0 # seconds 
predictT = 60.0 # seconds 
k = 20 # nearest neighbors 
maxdist = 30000.0 # meters
logpga = true # return predictins in log10(pga [cm/s^2]) 
jma_sort = false # apply JMA 0.3 second filtering 
grapes_preds = Array{GRAPES_pred}(undef, length(sample_times))
println("Running inference on Ridgecrest Earthquake")
for (ii, sample_time) in enumerate(sample_times)
    g, distance_from_earthquake, lon, lat = generate_graph(
        S, 
        rawT, 
        predictT, 
        ridge_loc, 
        sample_time, 
        k=k, 
        maxdist=maxdist, 
        logpga=logpga, 
        jma_sort=jma_sort,
    )
    out = model(g)
    grapes_preds[ii] = GRAPES_pred(
        g, 
        distance_from_earthquake, 
        lon, 
        lat, 
        Float64.(vec(out.ndata.x)), 
        sample_time, 
        origin_time,
    )
end

