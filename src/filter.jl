export kunugi_filt!, JMA_bandpass_filter

"""
kunugi_filt!(S)

Bandpass filter from Kunugi et al., 2013. 

Kunugi, T., Aoi, S., Nakamura, H., Suzuki, W., Morikawa, N., Fujiwara, H., 2013
An Improved Approximating Filter for Real-Time Calculation of Seismic Intensity,
Zisin (Journal of the Seismological Society of Japan. 2nd ser.), 2012, Volume 65, 
Issue 3, Pages 223-230,
https://www.jstage.jst.go.jp/article/zisin/65/3/65_223/_article/-char/en

Translated from C++ source code written by Yuki Kodera. 

# Arguments 
- `S::SeisData`: Seisdata
"""
function kunugi_filt!(S::SeisData)
    for ii in 1:S.n 
        kunugi_filt!(S[ii])
    end
    return nothing
end

function kunugi_filt!(C::SeisChannel)
    T = eltype(C.x)
    filter = Kunugi_Filter(2, 0.01, 6)
    old_x = zeros(T, 2, 6)
    old_y = zeros(T, 2, 6)

    # set initial filter values  
    for n in 1:filter.n 
        filtering_first!(filter, C.x[1], old_x, old_y, n)
    end

    # apply 6 2nd-order filters 
    new_y = zero(T)
    filters = 1:filter.n
    @inbounds for ii in eachindex(C.x)
        for n in filters 
            filtering_normal!(filter, new_y, C.x, ii, old_x, old_y, n)
        end
    end
    return nothing 
end

# Struct to store filter coeffiecients and recently filtered values 
struct Kunugi_Filter
    coeff_x::Matrix{Float64}
    coeff_y::Matrix{Float64}
    dim::Int 
    dt::Float64
    n::Int

    function Kunugi_Filter(
        coeff_x::Matrix{Float64},
        coeff_y::Matrix{Float64},
        dim::Int64,
        dt::Float64,
        n::Int64,
    )
        return new(coeff_x, coeff_y, dim, dt, n)
    end

    function Kunugi_Filter(dim, dt, n)
        coeff_x = zeros(dim+1, n)
        coeff_y = zeros(dim+1, n)

        # coefficients 
        f0 = 0.45	# Hz
        f1 = 7.0	# Hz
        f2 = 0.5	# Hz
        f3 = 12.0	# Hz
        f4 = 20.0	# Hz
        f5 = 30.0	# Hz
        h2a = 1.0
        h2b = 0.75
        h3 = 0.9
        h4 = 0.6
        h5 = 0.6
        g = 1.262

        # coefficient arrays 
        alpha = zeros(3)
        beta = zeros(3)

        # first filter 
        fa1 = f0 
        fa2 = f1 
        wa1 = 2 * π * fa1 
        wa2 = 2 * π * fa2 
        alpha[1] = (8.0 / (dt * dt)) + ((4.0 * wa1 + 2.0 * wa2) / dt) + wa1 * wa2
        alpha[2] = 2.0 * wa1 * wa2 - (16.0 / (dt*dt))
        alpha[3] = (8.0 / (dt * dt)) - ((4.0 * wa1 + 2.0 * wa2) / dt) + wa1 * wa2
        beta[1] = (4.0 / (dt * dt)) + (2.0 * wa2 / dt)
        beta[2] = -8.0 / (dt * dt)
        beta[3] = (4.0 / (dt * dt)) - (2.0 * wa2 / dt)
        gain = 1.0 
        for ii in 1:3 
            coeff_x[ii,1] = gain * beta[ii] / alpha[1]
            coeff_y[ii,1] = gain * alpha[ii] / alpha[1]
        end

        # second filter 
        fa3 = f1 
        wa3 = 2 * π * fa3 
        alpha[1] = (16.0 / (dt * dt)) + (17.0 * wa3 / dt) + wa3 * wa3
        alpha[2] = 2.0 * wa3 * wa3 - (32.0 / (dt * dt))
        alpha[3] = (16.0 / (dt * dt)) - (17.0 * wa3 / dt) + wa3*wa3
        beta[1] = (4.0 / (dt * dt)) + (8.5 * wa3 / dt) + wa3 * wa3
        beta[2] = 2.0 * wa3 * wa3 - (8.0 /(dt * dt))
        beta[3] = (4.0 / (dt * dt)) - (8.5 * wa3 / dt) + wa3 * wa3
        gain = 1.0
        for ii in 1:3 
            coeff_x[ii,2] = gain * beta[ii] / alpha[1]
            coeff_y[ii,2] = gain * alpha[ii] / alpha[1]
        end

        # third filter
        hb1 = h2a
        hb2 = h2b
        fb = f2
        wb = 2.0 * π * fb
        alpha[1] = (12.0 / (dt * dt)) + (12.0 * hb2 * wb / dt) + wb * wb
        alpha[2] = 10.0 * wb * wb - 24.0 / (dt * dt)
        alpha[3] = 12.0 / (dt * dt) - (12.0 * hb2 * wb / dt) + wb * wb
        beta[1]	= (12.0 / (dt * dt)) + (12.0 * hb1 * wb / dt) + wb * wb
        beta[2]	= 10.0 * wb * wb - (24.0 / (dt * dt))
        beta[3] = (12.0 / (dt * dt)) - (12.0 * hb1 * wb / dt) + wb * wb
        gain = 1.0
        for ii in 1:3 
            coeff_x[ii,3] = gain * beta[ii] / alpha[1]
            coeff_y[ii,3] = gain * alpha[ii] / alpha[1]
        end
                
        # fourth filter
        hc = h3
        fc = f3
        wc = 2.0* π * fc
        alpha[1] = (12.0 / (dt * dt)) + (12.0 * hc * wc / dt) + wc * wc
        alpha[2] = 10.0 * wc * wc - (24.0 / (dt * dt))
        alpha[3] = (12.0 / (dt * dt)) - (12.0 * hc * wc / dt) + wc * wc
        beta[1]	= wc * wc
        beta[2]	= 10.0 * wc * wc
        beta[3]	= wc * wc
        gain = 1
        for ii in 1:3 
            coeff_x[ii,4] = gain * beta[ii] / alpha[1]
            coeff_y[ii,4] = gain * alpha[ii] / alpha[1]
        end

        # fifth filter
        hc = h4
        fc = f4
        wc = 2.0 * π * fc
        alpha[1] = (12.0 / (dt * dt)) + (12.0 * hc * wc / dt) + wc * wc
        alpha[2] = 10.0 * wc * wc - (24.0 /(dt * dt))
        alpha[3] = (12.0 / (dt * dt)) - (12.0 * hc * wc / dt) + wc * wc
        beta[1]	= wc * wc
        beta[2]	= 10.0 * wc * wc
        beta[3]	= wc * wc
        gain = 1
        for ii in 1:3 
            coeff_x[ii,5] = gain * beta[ii] / alpha[1]
            coeff_y[ii,5] = gain * alpha[ii] / alpha[1]
        end
                
        # sixth filter
        hc = h5
        fc = f5
        wc = 2.0 * π * fc
        alpha[1] = (12.0 / (dt * dt)) + (12.0 * hc * wc / dt) + wc * wc
        alpha[2] = 10.0 * wc * wc - (24.0 / (dt * dt))
        alpha[3] = (12.0 / (dt * dt)) - (12.0 * hc * wc / dt) + wc * wc
        beta[1]	= wc * wc
        beta[2]	= 10.0 * wc * wc
        beta[3]	= wc * wc
        gain = g
        for ii in 1:3 
            coeff_x[ii,6] = gain * beta[ii] / alpha[1]
            coeff_y[ii,6] = gain * alpha[ii] / alpha[1]
        end

        # return filter object 
        return new(coeff_x, coeff_y, dim, dt, n)
    end
end

function filtering_first!(filter::Kunugi_Filter, new_x::Float32, old_x::Matrix, old_y::Matrix, n::Int)
    sumA = filter.coeff_x[1, n]
    sumB = 1.0 

    for ii in 2:filter.dim + 1 
        sumA += filter.coeff_x[ii, n]
        sumB += filter.coeff_y[ii, n]
    end

    temp_y = sumA * new_x / sumB 

    for ii in 1:filter.dim 
        old_x[ii, n] = new_x 
        old_y[ii, n] = temp_y 
    end
    return nothing  
end

function filtering_normal!(filter::Kunugi_Filter, new_y::Float32, x::Vector, ii::Int, old_x::Matrix, old_y::Matrix, n::Int)
    # apply IIR filter 
    # y[i] = a * x[n] + b * x[n - 1] + c * x[n - 2] + d * y[n - 1] + e * y[n - 2]
    new_y = filter.coeff_x[1,n] * x[ii] 
    new_y += filter.coeff_x[2,n] * old_x[2, n] 
    new_y -= filter.coeff_y[2, n] * old_y[2, n] 
    new_y += filter.coeff_x[3,n] * old_x[1, n] 
    new_y -= filter.coeff_y[3, n] * old_y[1, n]

    # update previous values 
    old_x[1, n] = old_x[2, n]
    old_y[1, n] = old_y[2, n]
    old_x[2, n] = x[ii]
    old_y[2, n] = new_y
    x[ii] = new_y
    return nothing
end 

"""
JMA_bandpass_filter(n, fs)

Bandpass filter from Karim & Yamazaki, 2002
Correlation of JMA instrumental seismic intensity with strong motion parameters
Earthquake Engng Struct. Dyn. 2002; 31:1191–1212 (DOI: 10.1002/eqe.158)
https://onlinelibrary.wiley.com/doi/abs/10.1002/eqe.158

# Arguments 
- `n::Integer`: Number of input sample points. 
- `fs::Real`: Sampling rate, in Hertz 

# Returns 
- `F::Vector`: Frequency-domain bandpass filter
"""
function JMA_bandpass_filter(n::Integer, fs::Real; f0::Real=0.5, fc::Real=10.0)
    freq = FFTW.rfftfreq(n, fs)
    F1 = sqrt.(1 ./ freq)
    F1[1] = F1[2] * 2
    x = sqrt.(freq ./ fc)
    F2 = @. (1 + 0.694 * x^2 + 0.241 * x^4 + 0.0557 * x^6 + 0.009664 * x^8 + 0.00134 * x^10 + 0.000155 * x^12) ^ -0.5
    F3 = @. sqrt(1 - exp((-freq / f0) ^ 3))
    return F1 .* F2 .* F3
end