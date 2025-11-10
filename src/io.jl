export read_kiknet_channel, load_kiknet, read_hypocenter, parse_hypocenter


"""

  load_kiknet(tarball)

Load KiK-net .kik.tar.gz file into SeisData and SeisHdr. 

Extracts .kik.tar.gz into temporary file, then loads all
surface level channels into SeisData struct. Returns details
about event magnitude as SeisIO.Quake.SeisHdr struct. 

# Arguments 
- `tarball`: Path to .kik.tar.gz file 
"""
function load_kiknet(tarball)

    # extract tarball to temporary directory 
    tar_gz = open(tarball)
    tar = GzipDecompressorStream(tar_gz)
    dir = Tar.extract(tar)
    close(tar)
    close(tar_gz)

    # read all surface acceleration files 
    tarfiles = readdir(dir,join=true) 
    surfacefiles = [f for f in tarfiles if any(endswith.(f, [".EW", ".NS", ".UD",".EW2", ".NS2", ".UD2"]))]
    S = SeisData() 
    _, mag = read_kiknet_channel(surfacefiles[1])
    if endswith(tarball, "knt.tar.gz")
        net = "KN"
    elseif endswith(tarball, "kik.tar.gz")
        net = "KI"
    else 
        net = "BO"
    end
    for sf in surfacefiles
        C, mag = read_kiknet_channel(sf, net=net)
        S += C 
    end

    # remove files 
    rm(dir, recursive=true)
    
    return S, mag 
end

"""

  read_kiknet_channel(file)

Load KiK-net channel file into SeisChannel and SeisHdr.

# Arguments 
- `file`: Path to Kik-net (.EW2, .NS2, .UD2) file 
"""
function read_kiknet_channel(file; net::String = "BO")
    channelstr = readlines(file)

    # read into SeisHdr struct 
    S = SeisHdr()

    # read meta-data line by line 
    S.ot = DateTime(channelstr[1][19:end],DateFormat("yyyy/mm/dd HH:MM:SS.ss"))
    S.loc.lat = parse(Float64, channelstr[2][19:end])
    S.loc.lon = parse(Float64, channelstr[3][19:end])
    S.loc.dep = parse(Float64, channelstr[4][19:end])
    S.mag.val = parse(Float64, channelstr[5][19:end])
    S.id = join([net,channelstr[6][19:end],"",split(file,".")[end]],".")

    # read into SeisChannel struct 
    C = SeisChannel()
    C.name = S.id
    C.id = S.id
    C.loc.lat = parse(Float64, channelstr[7][19:end])
    C.loc.lon = parse(Float64, channelstr[8][19:end])
    C.loc.el = parse(Float64, channelstr[9][19:end])
    starttime = DateTime(channelstr[10][19:end],DateFormat("yyyy/mm/dd HH:MM:SS.ss"))
    starttime -= Second(15)
    C.fs = parse(Float64, channelstr[11][19:end-2])
    Nsamples = (length(channelstr) - 17) * 8
    C.t = [[1 round(Int, datetime2unix(starttime) * 1e6)]; [Nsamples 0]]
    a,b = parse.(Int,split(replace(channelstr[14][19:end],"(gal)"=>""), "/")) 
    C.gain = 1.0  
    C.units = "cm/s^2"

    # load waveform data 
    C.x = zeros(Float32, Nsamples) 
    for ii in 18:length(channelstr)
        for jj in 1:length(channelstr[ii]) ÷ 9 
            C.x[(ii - 18) * 8 + jj] = parse(Float32, channelstr[ii][(jj - 1) * 9 + 1 : jj * 9])
        end
    end

    # check for samples at end 
    Nleftover = 8 - (length(channelstr[end]) ÷ 9)
    if Nleftover > 0 
        C.x[end - Nleftover + 1 : end] .= mean(C.x[1 : end - Nleftover])
    end

    # remove the gain 
    C.x .*= a / b 

    # remove the mean 
    demean!(C)
    return C, S 
end

"""

  read_hypocenter(file)

Read JMA Hypocenter record format into DataFrame 

Format is detailed at https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/format/hypfmt_e.html

# Argumnents 
- `file::String`: Path to hypocenter file
"""
function read_hypocenter(file)
    records = readlines(file)
    return vcat(parse_hypocenter.(records)...)
end

"""

  parse_hypocenter(record::String)

Parse JMA Hypocenter record format into DataFrame 

Format is detailed at https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/format/hypfmt_e.html

# Argumnents 
- `record::String`: 96 character string
"""
function parse_hypocenter(record::String)

    # read origin time 
    YR = parse(Int, record[2:5])         # year 
    MM = parse(Int, record[6:7])         # month 
    DD = parse(Int, record[8:9])         # day 
    hh = parse(Int, record[10:11])       # hour 
    mm = parse(Int, record[12:13])       # minute 
    SS = parse(Int, record[14:15])       # second 
    ss = parse(Int, record[16:17]) *  10 # millisecond
    ot = DateTime(YR, MM, DD, hh, mm, SS, ss)
    
    # error on origin time 
    SSerr = empty_parse(Float64, record[18:19])         # second 
    sserr = empty_parse(Float64, record[20:21]) / 100.0 # millisecond
    SSerr += sserr

    # location 
    latsign = record[22] == '-' ? -1 : 1 
    lat = empty_parse(Float64, record[23:24]) + parse(Float64, record[25:28]) / 100.0 / 60.0
    lat *= latsign 
    laterr = empty_parse(Float64, record[29:32]) / 10.0 / 60.0 
    lonsign = record[33] == '-' ? -1 : 1 
    lon = empty_parse(Float64, record[34:36]) + parse(Float64, record[37:40]) / 100.0 / 60.0
    lon *= lonsign 
    lonerr = empty_parse(Float64, record[41:44]) / 10.0 / 60.0
    depth = empty_parse(Float64, record[45:47]) 
    depth += empty_parse(Float64, record[48:49]) / 100.0 
    deptherr = empty_parse(Float64, record[50:52]) / 100.0

    # magnitude encoding is weird 
    # M 3.4  => 34 
    # M 1.2  => 12 
    # M 0.5  => 05
    # M -0.5 => -5
    # M -1.6 => A6 
    # M -2.8 => B8 
    # M -3.1 => C1 
    magcode = record[53:54]
    if !(magcode[1] in ['-', 'A', 'B', 'C', ' '])
        mag = parse(Float64, magcode) ./ 10.0 
    elseif magcode[1] == '-'
        mag = parse(Int, magcode[2]) ./ -10.0
    elseif magcode[1] == 'A'
        mag = -1.0 - parse(Int, magcode[2]) ./ -10.0
    elseif magcode[1] == 'B'
        mag = -2.0 - parse(Int, magcode[2]) ./ -10.0
    elseif magcode[1] == 'C'
        mag = -3.0 - parse(Int, magcode[2]) ./ -10.0
    else
        mag = missing 
    end

    # Travel time table 
    traveltime = record[59]

    # Hypocenter location precision
    hlp = record[60]

    # Subsidiary information
    si = record[61]

    # Maximum Intensity encoding is weird
    micode = record[62]
    if micode in ['0','1','2','3','4','7']
        mi = parse(Int, micode)
    elseif micode == 'A'
        mi = 4.75
    elseif micode == 'B'
        mi = 5.25
    elseif micode == 'C'
        mi = 5.75
    elseif micode == 'D'
        mi = 6.25
    else
        mi = missing
    end

    # Damage class 
    dc = record[63]

    # Tsunami class 
    tc = record[64]

    # District number 
    dn = empty_parse(Int, record[65])

    # Region number 
    rn = empty_parse(Int, record[66:68])

    # Region name 
    regionname = record[69:92]

    # Number of stations 
    nsta = empty_parse(Int, record[93:95])

    # Hypocenter determination flag
    hdf = record[96]

    return DataFrame(
        ORIGIN_DATETIME=ot,
        LON=lon,
        LAT=lat,
        DEPTH=depth,
        MAG=mag,
        ORIGIN_ERROR=SSerr,
        LON_ERROR=lonerr,
        LAT_ERROR=laterr,
        DEPTH_ERROR=deptherr,
        TRAVELTIME=traveltime,
        HYPOCENTER_LOCATION_PRECISION=hlp,
        MAXIMUM_INTENSITY=mi,
        SUBSIDIARY_INFO=si,
        DAMAGE_CLASS=dc,
        TSUNAMI_CLASS=tc,
        DISTRICT_NUMBER=dn,
        REGION_NUMBER=rn,
        REGION_NAME=regionname,
        NSTA=nsta,
        HYPOCENTER_DETERMINATION_FLAG=hdf
    )
    
end

function empty_parse(T, s)
    if s == ' ' ^ length(s)
        return zero(T)
    else
        return parse(T, s)
    end
end