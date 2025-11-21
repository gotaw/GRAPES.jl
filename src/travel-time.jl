export predictTT, JMA_travel_time

const _travel_time_cache = Dict{Tuple{Symbol,String},Any}()
const _travel_time_lock = ReentrantLock()

function _load_travel_time_interp(source::Symbol, phase::Union{AbstractString, AbstractChar})
    phase_tag = uppercase(String(phase))
    key = (source, phase_tag)
    lock(_travel_time_lock) do
        return get!(_travel_time_cache, key) do
            res_name = source === :JMA ? "JMA" : "AK135"
            path = joinpath(@__DIR__, "../resources/$(res_name)-$(phase_tag)-travel-time.jld2")
            JLD2.load(path)["$(phase_tag)_travel_time"]
        end
    end
end

"""

    JMA_travel_time(EQ, S)

Compute travel time from earthquake `EQ` to each station in `S`. 

Uses the Japan Meterological Agency travel time table for earthquake-station 
distances less than 2000 km or for earthquakes shallower than 700 km. 
Uses AK-135 travel time table, otherwise. 

# Arguments 
- `EQ::EQLoc`: Earthquake location (longitude, latitude and, depth)
- `S::SeisData`: Seisdata with station locations stored in S.loc

# Optional 
- `p::String`: Seismic phase to calculate travel time. "p" for P wave or "s" for S wave. Defaults to P wave.  

# Returns 
- `travel_times::Vector`: Travel times in seconds for each station in 'S' from earthquake 'EQ'. 
"""
function JMA_travel_time(EQ::EQLoc, S::SeisData, phase::Union{AbstractString, AbstractChar}="p")

    # load correct station table 
    jma_travel_time_interp = _load_travel_time_interp(:JMA, phase)
    ak135_travel_time_interp = _load_travel_time_interp(:AK135, phase)

    travel_times = zeros(S.n)
    # calculate travel time 
    for ii in eachindex(travel_times)
        distance_from_earthquake = Geodesics.surface_distance(
            EQ.lon,
            EQ.lat, 
            S.loc[ii].lon,
            S.loc[ii].lat,
            Geodesics.EARTH_R_MAJOR_WGS84,
        )
        distance_from_earthquake /= 1000
        # use JMA for depth less than 700km and distance less than 2000km
        if (distance_from_earthquake < 2000.0) && (EQ.dep < 700.0)
            travel_times[ii] = jma_travel_time_interp(EQ.dep, distance_from_earthquake)
        else # use ak135 
            travel_times[ii] = ak135_travel_time_interp(EQ.dep, distance_from_earthquake)
        end
    end

    return travel_times
end

function JMA_travel_time(EQ::EQLoc, ST::GeoLoc, phase::Union{AbstractString, AbstractChar}="p")
    # load correct station table 
    jma_travel_time_interp = _load_travel_time_interp(:JMA, phase)
    ak135_travel_time_interp = _load_travel_time_interp(:AK135, phase)

    # calculate Earthquake - station distance
    distance_from_earthquake = Geodesics.surface_distance(
        EQ.lon,
        EQ.lat, 
        ST.lon,
        ST.lat,
        Geodesics.EARTH_R_MAJOR_WGS84,
    )
    distance_from_earthquake /= 1000

    # use JMA for depth less than 700km and distance less than 2000km
    if (distance_from_earthquake < 2000.0) && (EQ.dep < 700.0)
        return jma_travel_time_interp(EQ.dep, distance_from_earthquake)
    else # use ak135 
        return ak135_travel_time_interp(EQ.dep, distance_from_earthquake)
    end
end

function JMA_travel_time(distance_from_earthquake::AbstractArray, earthquake_depth::Real, phase::Union{AbstractString, AbstractChar}="p")
    # load proper station table 
    jma_travel_time_interp = _load_travel_time_interp(:JMA, phase)
    ak135_travel_time_interp = _load_travel_time_interp(:AK135, phase)

    # calculate travel times
    travel_times = zero(distance_from_earthquake)
    for ii in eachindex(travel_times)

        # use JMA for depth less than 700km and distance less than 2000km
        if (distance_from_earthquake[ii] < 2000.0) && (earthquake_depth < 700.0)
            travel_times[ii] = jma_travel_time_interp(earthquake_depth, distance_from_earthquake[ii])
        else # use AK135
            travel_times[ii] = ak135_travel_time_interp(earthquake_depth, distance_from_earthquake[ii])
        end
    end

    return travel_times
end

"""

    predictTT(EQ, ST)

Compute travel time from earthquake `EQ` to station in `ST`. 

Uses AK-135 travel time table. 

# Arguments 
- `EQ::EQLoc`: Earthquake location (longitude, latitude and, depth)
- `ST::GeoLoc`: Seisdata with station locations stored in S.loc

# Optional 
- `p::String`: Seismic phase to calculate travel time. "p" for P wave or "s" for S wave. Defaults to P wave.  

# Returns 
- `travel_times::Float64`: Travel times in seconds for station 'ST' from earthquake 'EQ'.
"""
function predictTT(EQ::EQLoc, ST::GeoLoc, phase::Union{AbstractString, AbstractChar}="p")

    ak135_travel_time_interp = _load_travel_time_interp(:AK135, phase)

    # calculate Earthquake - station distance
    distance_from_earthquake = Geodesics.surface_distance(
        EQ.lon,
        EQ.lat, 
        ST.lon,
        ST.lat,
        Geodesics.EARTH_R_MAJOR_WGS84,
    )
    distance_from_earthquake /= 1000
    return ak135_travel_time_interp(EQ.dep, distance_from_earthquake)
end

function predictTT(EQ::EQLoc, S::SeisData, phase::Union{AbstractString, AbstractChar}="p")
    ak135_travel_time_interp = _load_travel_time_interp(:AK135, phase)

    travel_times = zeros(S.n)
    for ii in eachindex(travel_times)
        distance_from_earthquake = Geodesics.surface_distance(
            EQ.lon,
            EQ.lat, 
            S.loc[ii].lon,
            S.loc[ii].lat,
            Geodesics.EARTH_R_MAJOR_WGS84,
        )
        distance_from_earthquake /= 1000
        travel_times[ii] = ak135_travel_time_interp(EQ.dep, distance_from_earthquake)
    end
    return travel_times
end

function predictTT(
    event_longitude::Real, 
    event_latitude::Real, 
    event_depth::Real, 
    station_longitude::AbstractVector, 
    station_latitude::AbstractVector, 
    phase::Union{AbstractString, AbstractChar}="p",
)
    @assert length(station_latitude) == length(station_longitude)
    N = length(station_latitude)
    travel_times = zeros(N)
    ak135_travel_time_interp = _load_travel_time_interp(:AK135, phase)
    for ii in eachindex(travel_times)
        distance_from_earthquake = Geodesics.surface_distance(
            event_longitude,
            event_latitude, 
            station_longitude[ii],
            station_latitude[ii],
            Geodesics.EARTH_R_MAJOR_WGS84,
        )
        distance_from_earthquake /= 1000
        travel_times[ii] = ak135_travel_time_interp(event_depth, distance_from_earthquake)
    end
    return travel_times
end
