"""
This script downloads channels for the 2019 M7.1 Ridgecrest earthquake. 
You may have to run multiple times to download every channel - data centers 
can be finnicky at times...
"""

# function to download data for Ridgecrest 
function ridgecrest_download(channels_to_download, starttime, endtime)
    println("Downloading data from IRIS $(now())")
    IRISdata = get_data(
        "FDSN", 
        channels_to_download,
        src="IRIS", 
        s=starttime, 
        t=endtime,
        msr=false,
    )
    merge!(IRISdata)
    SCEDCchans = setdiff(channels_to_download, IRISdata.id)
    println("Downloading data from SCEDC $(now())")
    SCEDCdata = SeisData() 
    N = 3
    for ii in 1:N:length(SCEDCchans)
        idx = ii : min(ii + N - 1, length(SCEDCchans))
        SCEDCdata += get_data(
            "FDSN",
            SCEDCchans[idx],
            src="SCEDC",
            s=starttime, 
            t=endtime,
            msr=false,
            si=false,
        )
        # be nice to data center 
        sleep(0.1)
    end
    println("\nDownload Complete!\n")
    S = SeisData(IRISdata, SCEDCdata)
    return S
end

# download channels for 2019 M7.1 Ridgecrest earthquake
RIDGECREST_DIR = joinpath(@__DIR__,"../resources/Ridgecrest")  
channels = readlines(joinpath(RIDGECREST_DIR, "channels.txt")) 
N_channel = length(channels)
origin_time = DateTime(2019, 7, 6, 3, 19, 53)
starttime = origin_time - Second(15) # origin time - 15 seconds 
endtime = origin_time + Second(90) # origin time + 1.5 minutes 
date_format = DateFormat("yyyy.mm.dd.HH.SS")
date_str = Dates.format(starttime, date_format)

# check channels on device 
possible_files = [joinpath(download_cache, join([channel, date_str, "seisio"], '.')) for channel in channels]
downloaded_files = readdir(download_cache, join=true)
files_to_download = setdiff(possible_files, downloaded_files)
channels_to_download = [join(split(basename(c), '.')[1:4],'.') for c in files_to_download]

if !isempty(channels_to_download)

    # get channel information - split between IRIS and SCEDC based on network 
    provider = [channel[1:2] in ["NP", "PG", "WR"]  ? "SCEDC" : "IRIS" for channel in channels_to_download]
    IRIS_idx = findall(provider .== "IRIS")
    SCEDC_idx = findall(provider .== "SCEDC")
    println("Downloading instrument responses from IRIS $(now())")
    if !isempty(IRIS_idx) 
        IRISresp = FDSNsta(channels_to_download[IRIS_idx], s=starttime, t=endtime, msr=true, src="IRIS")
    else 
        IRISresp = SeisData()
    end
    println("Downloading instrument responses from SCEDC $(now())")
    if !isempty(SCEDC_idx)
        SCEDCresp = FDSNsta(channels_to_download[SCEDC_idx], s=starttime, t=endtime, msr=true, src="SCEDC")
    else
        SCEDCresp = SeisData()
    end
    resp = SeisData(IRISresp, SCEDCresp)

    # update station response time 
    for ii in 1:resp.n 
        s = resp.misc[ii]["startDate"]
        e = resp.misc[ii]["endDate"]
        L = round(Int, (e - s ) * 1e-6 * resp.fs[ii])
        resp.t[ii] = Array{Int64,2}([[1 s];[L 0]])
    end

    # download data - chunk the SCEDC request 
    S = ridgecrest_download(channels_to_download, starttime, endtime)

    if !isempty(S)

        # add instrument responses 
        add_response!(S, resp)

        # write to disk 
        for idx in 1:S.n 
            outpath = joinpath(download_cache, join([S.id[idx], date_str, "seisio"], '.'))
            wseis(outpath, S[idx])
        end
    end
end

# remove dataless files 
files = readdir(download_cache, join=true)
file_sizes = filesize.(files)
remove_idx = findall(file_sizes .< 1000) # bytes 
rm.(files[remove_idx])

# remove xml file from download
if isfile("FDSNsta.xml")
    rm("FDSNsta.xml") 
end
