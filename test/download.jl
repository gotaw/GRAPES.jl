"""
This script downloads channels for the 2019 M7.1 Ridgecrest earthquake. 
You may have to run multiple times to download every channel - data centers 
can be finnicky at times...
"""

# download channels for 2019 M7.1 Ridgecrest earthquake
RIDGECREST_DIR = joinpath(@__DIR__,"../resources/Ridgecrest")  
channels = readlines(joinpath(RIDGECREST_DIR, "channels.txt")) 
origin_time = DateTime(2019, 7, 6, 3, 19, 53)
starttime = origin_time - Second(15) # origin time - 15 seconds 
endtime = origin_time + Second(90) # origin time + 1.5 minutes 

# download data 
N_channel = length(channels)
date_format = DateFormat("yyyy.mm.dd.HH.SS")
date_str = Dates.format(starttime, date_format)
for (idx, channel) in enumerate(channels) 
    print("Downloading $channel, $idx of $N_channel $(now())\r")
    outpath = joinpath(download_cache, join([channel, date_str, "seisio"], '.'))
    wave_src = channel[1:2] in ["AZ", "NN", "SN", "YN"]  ? "IRIS" : "SCEDC"
    gain_src = channel[1:2] in ["NP", "PG", "WR"]  ? "SCEDC" : "IRIS"
    if !isfile(outpath)
        try 
            C = get_data(
                    "FDSN", 
                    channel,
                    src=wave_src, 
                    s=starttime, 
                    t=endtime,
                    msr=true,
            )
            R = FDSNsta(channel, s=starttime, t=endtime, msr=true, src=gain_src)
            merge!(C)

            # update station response time 
            for ii in 1:R.n 
                s = R.misc[ii]["startDate"]
                e = R.misc[ii]["endDate"]
                L = round(Int, (e - s ) * 1e-6 * R.fs[ii])
                R.t[ii] = Array{Int64,2}([[1 s];[L 0]])
            end
            add_response!(C, R)

            # write to disk 
            wseis(outpath, C)
        catch 
            @warn "No data available for $channel"
            continue 
        end
        # be nice to the data center 
        sleep(0.1)
    end
end
println("\nDownload Complete!\n")

# remove dataless files 
files = readdir(download_cache, join=true)
file_sizes = filesize.(files)
remove_idx = findall(file_sizes .< 1000) # bytes 
rm.(files[remove_idx])

# remove xml file from download
if isfile("FDSNsta.xml")
    rm("FDSNsta.xml") 
end
