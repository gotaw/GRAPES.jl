export smooth, smooth!, envelope, envelope!, update_time!, add_response!, moving_abs_maximum

"""

    smooth(x, n)

Smooth vector `x` with `n` point moving mean. 

# Arguments 
- `x::AbstractVector`: time-series
- `n::Integer`: number of sample points for moving mean
"""
smooth(x::AbstractVector{T}, n::Integer) where T <: AbstractFloat = DSP.conv(x, fill(T(1 / n), n))[n ÷ 2 + isodd(n)  : end - n ÷ 2]
smooth!(x, n::Integer) = x .= smooth(x,n)

"""

    moving_abs_maximum(S, n)

Simple moving absolute maximum of channels in SeisData `S` of `n` points. 

# Arguments 
- `S::SeisData`:
- `n::Integer`: number of sample points for moving mean
"""
function moving_abs_maximum(S::SeisData, n::Integer)
    Smax = deepcopy(S)
    for ii in eachindex(S.x)
        Smax.x[ii] .= moving_abs_maximum(S.x[ii], n)
    end
    return Smax
end

function moving_abs_maximum(C::SeisChannel, n::Integer)
    Cmax = deepcopy(C)
    Cmax.x .= moving_abs_maximum(C.x, n)
    return Cmax
end

function moving_abs_maximum(x::AbstractVector, n::Integer)
    x = abs.(x)
    L = length(x)
    xmax = zeros(eltype(x), L)
    n21 = 2 * n + 1
    for ii in eachindex(x)[1:n]
        b = @view x[1:ii+n]
        xmax[ii] = sum(b) / length(b)
    end
    for ii in eachindex(x)[end-n+1:end]
        b = @view x[ii - n:L]
        xmax[ii] = sum(b) / n21
    end
    for ii in eachindex(x)[1+n:end-n]
        b = @view x[ii-n:ii+n]
        xmax[ii] = sum(b) / length(b)
    end
    return xmax
end

"""

    envelope(A)

Envelope of `A` along the first dimension using the Hilbert Transform. 

# Arguments 
- `A::AbstractArray`: time-series data 
"""
envelope(A::AbstractArray) = abs.(hilbert(A))
envelope!(A::AbstractArray) = A .= envelope(A)

"""

    envelope(S)

Envelope of channels in SeisData `S` using the Hilbert Transform. 

# Arguments 
- `S::SeisData`
"""
function envelope!(S::SeisData)
    for ii in eachindex(S.x)
        S.x[ii] .= envelope(S.x[ii])
    end
    return nothing 
end
envelope(S::SeisData) = (
    Senv = deepcopy(S);
    envelope!(Senv);
    return Senv
)
function envelope!(C::SeisChannel)
    envelope!(C.x)
    return nothing 
end
envelope(C::SeisChannel) = (
    Cenv = deepcopy(C);
    envelope!(Cenv);
    return Cenv
)

"""

    update_time!(S)

Update S.t for station response. 

"""
function update_time!(S::SeisData)
    for ii in 1:S.n
        S.t[ii] = [[1 S.misc[ii]["startDate"]]; [(Int(S.misc[ii]["endDate"] - S.misc[ii]["startDate"]) / 1e6 * S.fs[ii]) 0]]
    end
    return nothing 
end

"""

    add_response!(S)

Update S.t for station response. 
# Arguments 
- `S::SeisData`: `SeisData` with event waveforms. 
- `M::SeisData`: `SeisData` with station response.

"""
function add_response!(S::SeisData, M::SeisData)
    todelete = Int[]
    for ii in 1:S.n
        ind = findall(M.id .== S.id[ii])
        isempty(ind) && append!(todelete, ii) && println("No response $(S.id[ii])") && continue
       
        # get starttime 
        s = S.t[ii][1,2]
        starttimes = [M.t[ind][jj][1,2] for jj in 1:length(ind)]
        endtimes = [starttimes[jj] .+ round(Int,M.t[ind][jj][2,1] ./ M.fs[ind][jj] .* 1e6) for jj in 1:length(ind)]
        jj = ind[findfirst(starttimes .< s .< endtimes)]
        isnothing(jj) && append!(todelete, ii) && println("No response $(S.id[ii])") && continue
        S.loc[ii] = M.loc[jj]
        S.gain[ii] = M.gain[jj]
        S.resp[ii] = M.resp[jj]
        S.units[ii] = M.units[jj]
    end
    return nothing
end