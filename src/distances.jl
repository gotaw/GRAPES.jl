export distance, adjacency, distance_threshold!, weighted_distance!, weighted_distance, knn_adjacency
export rewire!, rewire, shortcut!, shortcut, kleinberg_navigation!, kleinberg_navigation
export kleinberg_shortcut!, kleinberg_shortcut, reverse_distance!, reverse_distance
export boore_atkinson, fukushima_tanaka, fukushima_tanaka_amplitude, kanno_2006, kanno_2006_amplitude

"""
  distance(lat, lon)

Euclidean distance between all points in vectors `lat`, `lon` and `elev`. 

Returns a symmetric matrix with distances between all points in meters. 

# Arguments 
- `lat::AbstractVector`: Vector of latitudes.
- `lon::AbstractVector`: Vector of longitudes. 
- `elev::AbstractVector`: Vector of elevations in meters. 
"""
function distance(lat::AbstractVector, lon::AbstractVector)
    # check that input is in lat, lon
    any(abs.(lat) .> 90.0) && throw(ArgumentError("Inputs must be in latitude, longitude, elevation order!"))
    length(lon) != length(lat) && throw(ArgumentError("Latitude and longitude vectors must be of same length"))
    N = length(lat)
    T = eltype(lat)
    dist = zeros(T,N,N)
    for ii = 1:N
        for jj = ii:N
            dist[jj,ii] = Geodesics.surface_distance(lon[ii], lat[ii], lon[jj], lat[jj], Geodesics.EARTH_R_MAJOR_WGS84)
            dist[ii,jj] = dist[jj,ii]
        end
    end
    return dist
end
distance(lat::AbstractVector, lon::AbstractVector, elev::AbstractVector) = distance(lat, lon)
function distance(L::Vector{T}) where T <: InstrumentPosition
    lat = [L[ii].lat for ii in eachindex(L)]
    lon = [L[ii].lon for ii in eachindex(L)]
    return distance(lat, lon)
end

"""
  distance(lat, lon, EQ)

Euclidean distance between all points in vectors `lat`, `lon` and an earthquake location `EQ`. 

Returns a vector of distances between stations and earthquake in meters. 

# Arguments 
- `lat::AbstractVector`: Vector of latitudes.
- `lon::AbstractVector`: Vector of longitudes. 
- `EQ::EQLoc`: QuakeML-compliant earthquake location. 
"""
function distance(lat::AbstractVector, lon::AbstractVector, EQ::EQLoc)
    # check that input is in lat, lon, elev form 
    any(abs.(lat) .> 90.0) && throw(ArgumentError("Inputs must be in latitude, longitude, EQLoc order!"))
    length(lon) != length(lat) && throw(ArgumentError("Latitude and longitude vectors must be of same length"))
    N = length(lat)
    T = eltype(lat)
    dist = zeros(T,N)
    for ii = 1:N
            dist[ii] = Geodesics.surface_distance(lon[ii], lat[ii], EQ.lon, EQ.lat, Geodesics.EARTH_R_MAJOR_WGS84)
    end
    return dist
end

"""

  boore_atkinson_amplitude(distance)

Relative seismic amplitude with distance. 

Relation taken from Figure 6b of: 

Boore, D. M., & Atkinson, G. M. (2008). Ground-motion prediction equations for 
the average horizontal component of PGA, PGV, and 5%-damped PSA at spectral 
periods between 0.01 s and 10.0 s. Earthquake spectra, 24(1), 99-138.
https://journals.sagepub.com/doi/abs/10.1193/1.2830434

# Arguments
- `distance`: Joyner-Boore distance (in kilometers).

"""
@. function boore_atkinson_amplitude(distance) 
    return 0.885 * exp(-distance * 8.885e-2) .+ -7.329e-4 .*  distance + 0.158
end

@. function fukushima_tanaka(R, M)
    return 0.41 * M - log10(R + 0.032 * 10 ^ (0.41 * M)) - 0.0034 * R + 1.3
end

@. function kanno_2006(R, M)
    return 0.56 * M - log10(R + 0.0055 * 10 ^ (0.5 * M)) - 0.0031 * R + 0.26 + 0.37
end

"""

  fukushima_tanaka_amplitude(distance, magnitude)

Relative seismic amplitude as function of distance from source and surface-wave mangitude. 

Given by formula 

log10(A) = 0.41 * M - log10(R + 0.032 * 10 ^ (0.41 * M)) - 0.0034 * R + 1.3

Returns amplitude of surface-wave relative to surface-wave at distance = 0 km. 

Equation 15 from: 

Fukushima, Y., & Tanaka, T. (1990). A new attenuation relation for peak 
horizontal acceleration of strong earthquake ground motion in Japan. Bulletin
of the seismological Society of America, 80(4), 757-783.
https://pubs.geoscienceworld.org/ssa/bssa/article/80/4/757/102395/A-new-attenuation-relation-for-peak-horizontal

# Arguments
- `distance`: Joyner-Boore distance (in kilometers).
- `magnitude`: Surface-wave magnitude. 

"""
@. function fukushima_tanaka_amplitude(distance, M)
    return fukushima_tanaka(distance, M) / fukushima_tanaka(0, M)
end

"""

  kanno_2006_amplitude(distance, magnitude)

Relative seismic amplitude as function of distance from source and surface-wave mangitude. 

Given by formula 

log10(A) = 0.56 * M - log10(R + 0.0055 * 10 ^ (0.5 * M)) - 0.0031 * R + 0.26 + 0.37

Returns amplitude of surface-wave relative to surface-wave at distance = 0 km. 

Equation 5 from: 

Kanno, T., Narita, A., Morikawa, N., Fujiwara, H., & Fukushima, Y. (2006). A new 
attenuation relation for strong ground motion in Japan based on recorded data. 
Bulletin of the Seismological Society of America, 96(3), 879-897.
https://pubs.geoscienceworld.org/ssa/bssa/article/96/3/879/146775/A-New-Attenuation-Relation-for-Strong-Ground

# Arguments
- `distance`: Joyner-Boore distance (in kilometers).
- `magnitude`: Surface-wave magnitude. 

"""
@. function kanno_2006_amplitude(distance, M)
    return kanno_2006(distance, M) / kanno_2006(0, M)
end

"""
  distance_threshold!(dist, thresh_dist)

Set all values of `dist` >= `thresh_dist` to zero. 

# Arguments
- `dist::AbstractMatrix`: Distance matrix between all stations 
- `thresh_dist::Real`: Maximum distance between adjacent stations
"""
function distance_threshold!(dist::AbstractMatrix{T}, thresh_dist::Real) where T <: Real
    dist[findall(dist .>= thresh_dist)] .= T(0)
    return nothing
end
distance_threshold(dist::AbstractMatrix, thresh_dist::Real) = (
    A = deepcopy(dist);
    distance_threshold!(dist, thresh_dist);
    return A;
)

"""

  adjacency(dist,thresh_dist)

Adjacency matrix for distance matrix `dist`. 

Returns `dist[ii,jj] / thresh_dist` if distance between stations `ii` and `jj` is less than
`thresh_dist` and `0` otherwise. 

# Arguments
- `dist::AbstractMatrix`: Distance matrix between all stations 
- `thresh_dist::Real`: Maximum distance between adjacent stations
"""
function adjacency(dist::AbstractMatrix, thresh_dist::Real)
    T = eltype(dist)
    Nrows, Ncols = size(dist)
    ind = findall(dist .<= thresh_dist)
    adj = spzeros(T,size(dist,1),size(dist,2))
    adj[ind] .= dist[ind] ./ thresh_dist 
    return adj 
end
function adjacency(dist::AbstractMatrix)
    T = eltype(dist)
    Nrows, Ncols = size(dist)
    ind = findall(.!iszero.(dist))
    adj = spzeros(T,size(dist,1),size(dist,2))
    adj[ind] .= dist[ind]
    adj.nzval ./= maximum(adj.nzval)
    return adj 
end
function adjacency(g::AbstractGraph, dist::AbstractMatrix)
    T = eltype(dist)
    Nrows, Ncols = size(dist)
    @assert nv(g) == Nrows == Ncols "Number of nodes in graph must match size of distance matrix!"
    adj = zeros(T,size(dist,1),size(dist,2))

    # use incoming edges if directed
    if is_directed(g)
        for ii in 1:nv(g)
            adj[g.badjlist[ii],ii] = dist[g.badjlist[ii],ii]
        end
    else # use outgoing edges 
        for ii in 1:nv(g)
            adj[ii,g.fadjlist[ii]] = dist[ii,g.fadjlist[ii]]
        end
    end
    return sparse(adj) 
end

"""

  knn_adjacency(dist,k)

k-nearest neighbors adjacency matrix for distance matrix `dist`. 

Returns `dist[ii,jj]` if station `ii` is the `k`th the or less nearest 
neighbor to station `jj` and `0` otherwise. Distances are normalized 
by the largest `k` neighbor in `dist`. 

# Arguments
- `dist::AbstractMatrix`: Distance matrix between all stations 
- `k::Integer`: Number of neighbors k 
"""
function knn_adjacency(dist::AbstractMatrix, k::Integer)
    T = eltype(dist)
    Nrows, Ncols = size(dist)
    @assert k < Nrows 
    adj = spzeros(T,Nrows,Ncols)
    # iterate over columns 
    for ii = 1:Ncols
        ind = sort(sortperm(dist[:,ii])[2:k+1])
        adj[ind,ii] .= dist[ind,ii] 
    end
    adj.nzval ./= maximum(adj.nzval)
    return adj
end

"""

  weighted_distance(A,δ)

Exponentially distributed distance matrix. 

Apply `exp(-(A^2 / δ))`. Values close to 0 -> 1 and vice-versa. δ controls the sharpness of the scaling. 

# Arguments
- `A::AbstractArray`: Distance matrix between all linked nodes 
- `δ::Real`: Scaling paramter 
"""
function weighted_distance!(A::AbstractArray, δ::Real)
    A .= exp.(- (A .^ 2 ./ δ) )
    return nothing
end

function weighted_distance!(A::AbstractSparseArray, δ::Real)
    weighted_distance!(A.nzval, δ)
    return nothing 
end

function weighted_distance(A::AbstractArray, δ::Real)
    B = deepcopy(A)
    weighted_distance!(B, δ)
    return B
end

function weighted_distance(A::AbstractSparseArray, δ::Real)
    B = deepcopy(A)
    weighted_distance!(B, δ)
    return B
end

"""

  power_distance!(A, p)

Power-law distance matrix: returns `A ^ -p`.

# Arguments
- `A::AbstractArray`: Distance matrix between all linked nodes 
- `p::Real`
"""
function power_distance!(A::AbstractArray, p::Real)
    T = eltype(A)
    ind = .!iszero.(A)
    A[ind] .= A[ind] .^ -p
    A ./= maximum(abs.(A))
    return nothing 
end
power_distance!(A::AbstractSparseArray, p::Real) = power_distance!(A.nzval, p)
function power_distance(A::AbstractArray, p::Real)
    B = deepcopy(A)
    power_distance!(B, p)
    return B 
end
function power_distance(A::AbstractSparseArray, p::Real)
    B = deepcopy(A)
    power_distance!(B, p)
    return B 
end

"""

  reverse_distance!(A)

Normalizes `A` from 0 to 1, then returns 1 - `A`.

# Arguments
- `A::AbstractArray`: Distance matrix between all linked nodes 
"""
function reverse_distance!(A::AbstractArray)
    T = eltype(A)
    A ./= maximum(abs.(A))
    A .= one(T) .- A
    return nothing 
end
reverse_distance!(A::AbstractSparseArray) = reverse_distance!(A.nzval)
function reverse_distance(A::AbstractArray)
    B = deepcopy(A)
    reverse_distance!(B)
    return B 
end
function reverse_distance(A::AbstractSparseArray)
    B = deepcopy(A)
    reverse_distance!(B)
    return B 
end

"""
  rewire!(g, p)

Perform Watts-Strogatz rewiring to turn a large-world graph into a small-world. 

The algorithm proceeds as follows:
-  for a graph with on average `n` vertices and `e` edges: 
1. Consider each edge `e` in turn with its source `s` and destination `d`
2. Generate a uniformly random number `r`. If `p ≥ r`, then the edge `(s, d)` is
    left unaltered. Otherwise, the edge is deleted and *rewired* so that `s` is
    connected to some vertex `t`, chosen uniformly at random from the entire
    graph, excluding `s` and its neighbors. (Note that `t` is a valid candidate.)
For `p = 0`, the graph will remain unchanged, and for `p = 1`, all edges will
be rewired randomly.

# Arguments
`g::AbstractGraph`: Input graph
`p::Real`: Probability for rewiring

### Optional Arguments
- `seed=-1`: set the RNG seed.
"""
function rewire!(g::AbstractGraph, p::Real;seed::Int=-1)
    rng = Graphs.getRNG(seed)
    @assert zero(p) <= p <= one(p) "Probability p for rewiring must be in [0,1]!"

    # go through each vertex 
    n = nv(g)
    for e in collect(edges(g))
        if rand(rng) < p 
            while true
                d = rand(1:n)                     # Tentative new target
                d == e.src && continue            # Self-loops prohibited
                d == e.dst && break               # Rewired to original target
                has_edge(g, e.src, d) && continue # Already has this edge 
                add_edge!(g, e.src, d)            # Add new edge 
                rem_edge!(g, e.src, e.dst)        # Delete original edge
                break                             # We found a valid target
            end
        end
    end
    return nothing
end
function rewire(g::AbstractGraph, p::Real; seed::Int=-1)
        newg = deepcopy(g)
        rewire!(newg, p, seed=seed)
        return newg
end

"""
  shortcut!(g, p)

Add random edges to graph `g` with probability `p`.

Transforms `g` from a large-world graph into a small-world graph by adding random edges
following the algorithm of Newman, Moore, and Watts, 2000. 

The algorithm proceeds as follows:
-  for a graph with `n` vertices and `e` edges: 
1. Consider each `e` new edges
2. Generate a uniformly random number `r`. If `p ≥ r`, then the edge `(s, d)` is
    left unaltered. Otherwise, the a new edge is added so that `s` is
    connected to some vertex `t`, chosen uniformly at random from the entire
    graph, excluding `s` and its neighbors. (Note that `t` is a valid candidate.)
For `p = 0`, the graph will remain unchanged, and for `p = 1`, the graph will have `e`
new random edges.

# Arguments
`g::AbstractGraph`: Input graph
`p::Real`: Probability for adding shortcut

### Optional Arguments
- `seed=-1`: set the RNG seed.
"""
function shortcut!(g::AbstractGraph, p::Real; seed::Integer=-1)
    rng = Graphs.getRNG(seed)
    @assert zero(p) <= p <= one(p) "Probability p for new edge must be in [0,1]!"

    # go through each edge 
    nV = nv(g)
    nE = ne(g)
    for _ in 1:nE
        # limit to 5 shortcut attempts to avoid infinite loop
        if rand(rng) < p 
            ii = 0 
            while ii < 5
                ii += 1
                src = rand(1:nV)                     # Tentative new source
                dst = rand(1:nV)                     # tentative new destination
                src == dst && continue              # Self-loops prohibited
                has_edge(g, src, dst) && continue   # Already has this edge 
                add_edge!(g, src, dst) && break     # Add new edge 
            end
        end
    end
    return nothing
end
function shortcut(g::AbstractGraph, p::Real; seed::Int=-1)
    newg = deepcopy(g)
    shortcut!(newg, p, seed=seed)
    return newg
end

"""
kleinberg_navigation!(g, α, q)

Add `q` long-range edges per vertex to graph `g` with probability `r ^ -α`.

Transforms `g` from a large-world graph into a small-world graph by edges
following the algorithm of Kleinberg, 2000. 

The algorithm proceeds as follows:
-  for a graph with `n` vertices: 
1. Consider each vertex `v`
2. Calculate the molecular distance `r` to every other vertex. Select `q` new edges, each with 
probability `r ^ -α`. 

# Arguments
`g::AbstractGraph`: Input graph
`α::Real`: Exponent for distance calculation 
`q::Integer`: Number of new edges per vertex 

### Optional Arguments
- `seed=-1`: set the RNG seed.
"""
function kleinberg_navigation!(g::AbstractGraph, α::Real, q::Integer; seed::Integer=-1)
    rng = Graphs.getRNG(seed)
    @assert q <= nv(g) "Number of new edges must be less than number of nodes!"
    @assert α > 0 "Weight α must be greater than zero"

    # loop through each node 
    Nv = nv(g)
    verts = collect(vertices(g))
    for dst in 1:Nv
        # calculate shortest path to each node 
        ds = bellman_ford_shortest_paths(g, dst)

        # calculate probability weights
        r = Float64.(ds.dists) .^ -α
        r[r .>= 1.0] .= zero(eltype(r))
        w = Weights(r)

        # choose new edge 
        for ii in 1:q 
            src = sample(w)
            add_edge!(g, src, dst)
        end
    end
    return nothing 
end

function kleinberg_navigation(g::AbstractGraph, α::Real, q::Integer; seed::Integer=-1)
    newg = deepcopy(g)
    kleinberg_navigation!(newg, α, q, seed=seed)
    return newg
end

"""
kleinberg_shortcut!(g, α, p)

Add long-range edges to graph `g` with probability `p`.

Transforms `g` from a large-world graph into a small-world graph by edges
following the algorithm of Kleinberg, 2000. 

The algorithm proceeds as follows:
-  for a graph with `n` vertices and average degree `k`: 
1. Consider `s = n * k * p` random vertices 
2. Calculate the molecular distance `r` to every other vertex.
3. Select a new random edge from the probability distribution `r ^ -α`. 

# Arguments
`g::AbstractGraph`: Input graph
`α::Real`: Exponent for distance calculation 
`q::Integer`: Number of new edges per vertex 

### Optional Arguments
- `seed=-1`: set the RNG seed.
"""
function kleinberg_shortcut!(g::AbstractGraph, α::Real, p::Real; seed::Integer=-1)
    rng = Graphs.getRNG(seed)
    @assert α > 0 "Weight α must be greater than zero"
    @assert zero(p) <= p <= one(p) "Probability p for new edge must be in [0,1]!"

    # get number of shortcuts 
    Nv = nv(g)
    k = degree(g) |> mean 
    Nshort = round(Int, Nv * k * p)  

    # add new edges 
    dests = rand(1:Nv, Nshort)
    for dst in dests
        # calculate shortest path to each node 
        ds = bellman_ford_shortest_paths(g, dst)

        # calculate probability weights
        r = Float64.(ds.dists) .^ -α
        r[r .>= 1.0] .= zero(eltype(r))
        w = Weights(r)

        # choose new edge 
        src = sample(w)
        add_edge!(g, src, dst)
    end
    return nothing 
end

function kleinberg_shortcut(g::AbstractGraph, α::Real, p::Real; seed::Integer=-1)
    newg = deepcopy(g)
    kleinberg_shortcut!(newg, α, p, seed=seed)
    return newg
end
