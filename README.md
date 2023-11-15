# GRAPES.jl: GRAph Prediction of Earthquake Shaking 

GRAPES.jl is a Julia-language code for earthquake early warning. 

Author: Tim Clements, U.S. Geological Survey | tclements@usgs.gov

## Overview

GRAph Prediction of Earthquake Shaking (GRAPES) is an earthquake early warning (EEW) algorithm based on graph neural networks. GRAPES.jl implements GRAPES in the Julia language using the Flux.jl and GraphNeuralNetwork.jl deep learning packages. The GRAPES.jl package provides code for EEW ground motion predictions. 

## Getting Started

First [install Julia](https://julialang.org/downloads/) on your computer. We recommend using the latest version of Julia. To install GRAPES.jl, activate the package manager by typing `]`

```julia
pkg> add https://code.usgs.gov/esc/grapes.jl.git
```

We also recommend installing the following packages: 

```julia
pkg> add Graphs, GraphNeuralNetworks, Flux, Scratch 
pkg> add https://github.com/tclements/SeisIO.jl.git
```

If you intend to use GRAPES.jl on the GPU, we recommend using CUDA.jl: 

```
pkg> add CUDA 
```

## Using and testing GRAPES.jl 

The main product of GRAPES.jl is a trained `GRAPES_model` which predicts future shaking given a `GNNGraph` as input. To load a `GRAPES_model` startup the [REPL](https://docs.julialang.org/en/v1/stdlib/REPL/): 

```julia
julia> using GRAPES 
julia> model = load_GRAPES_model()
``````

The input to a `GRAPES_model` is a [`GNNGraph` from the GraphNeuralNetworks](https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/api/gnngraph/#GraphNeuralNetworks.GNNGraphs.GNNGraph).jl package. To create a `GNNGraph`, we recommend running the test suite, which downloads waveforms from the 2019 M7.1 Ridgecrest earthquake. First activate the package manager by typing `]`: 

```julia
pkg> test GRAPES 
```

This will download waveforms from Ridgecrest to the GRAPES scratchspace. This may take a few minutes! You can then create a GNNGraph by loading data from Ridgecrest using `SeisIO`: 

```julia
julia> using Dates, Scratch, SeisIO
julia> processed_cache = get_scratch!(GRAPES, "processed_files")
julia> waveform_path = joinpath(processed_cache, "SCSN.seisio")
julia> S = read_data(waveform_path)
```

GRAPES provides the `generate_graph` function to convert a `SeisIO.SeisData` structure into a `GNNGraph`. Here is an exmaple: 

```julia
# source parameters for Ridgecrest 
julia> origin_time = DateTime(2019, 7, 6, 3, 19, 53)
julia> sample_time = origin_time + Second(6)
julia> ridge_loc = EQLoc(lat=35.77, lon = -117.599, dep=8.0)

# convert to seismic waveform graph 
julia> rawT = 4.0 # seconds of input window
julia> predictT = 60.0 # seconds to extract future PGA 
julia> k = 20 # nearest neighbors 
julia> maxdist = 30000.0 # meters
julia> logpga = true # return true PGA in log10(pga [cm/s^2]) 
julia> g, distance_from_earthquake, lon, lat = generate_graph(
        S, 
        rawT, 
        predictT, 
        ridge_loc, 
        sample_time, 
        k=k, 
        maxdist=maxdist, 
        logpga=logpga, 
    )
```
This returns `g`, a `GNNGraph`, and three vectors, `distance_from_earthquake`, `lon`, `lat`, which denote the distance from the Ridgecrest, longitude and latitude of each station in SeisData `S`, respectively. Let's have a look at our seismic `GNNGraph` `g`: 

```julia
julia> g
GNNGraph:
  num_nodes: 341
  num_edges: 7177
  ndata:
        x = 400×3×1×341 Array{Float32, 4}
  gdata:
        u = 341×1 Matrix{Float32}
```

`g` has 341 nodes (stations) and 7177 edges (connections between stations). Waveform data from the 4 second window ending at DateTime(2019, 7, 6, 3, 19, 59) is stored in `g.data.x`. Note that the shape of `g.ndata.x` is `400x3x1x341`, with 400 coming from 4 seconds of data @ 100 Hz and 3 = three channel (east, north and vertical). Future peak ground acceleration (PGA) for each station is stored in g.gdata.u. GRAPES uses units of cm/s^2 in log10-space. To make a GRAPES prediction, call our `model` on `g`: 

```julia
julia> prediction = model(g)
```

`prediction` is also a `GNNGraph`. GRAPES's prediction error can be easily compared to true PGA in log10(cm/s^2) by: 

```julia
vec(prediction.ndata.x) .- vec(g.gdata.u)
```

For more detailed use of GRAPES.jl, we suggest you check out the example in the test set in /test!

## Contents of this Repository

- /src/: Source files for GRAPES.jl. 
- /test/: Test suite for GRAPES.jl 
- /resources/: Travel time files and models used in GRAPES.   
- Project.toml: The project file describes the project on a high level, for example, the package/project dependencies and compatibility constraints are listed in the project file. 
- Manifest.toml: The manifest file is an absolute record of the state of the packages in the environment. It includes exact information about (direct and indirect) dependencies of the project. 
- code.json: Includes metadata about this project for USGS software inventory.
- DISCLAIMER.md: Indicates the project status as a
  preliminary/provisional software release.
- LICENSE.md: Includes appropriate licensing information for this project and
  its dependencies.
- README.md: This file. Serves as a landing/introduction page for this project.
- CODE_OF_CONDUCT.md: Re-iterates USGS scientific code of  conduct to be
  following when contributing to the associated project.
- CONTRIBUTING.md: Describes the process for collaborators to contribute to
  this project.

## Citing 
If you use GRAPES.jl, please cite the following article: 

> Clements, T., Cochran, E., Baltay, A., Minson, S., Yoon, C., (2023), "GRAPES: Earthquake Early Warning by Passing Seismic Vectors Through the Grapevine", submitted to Geophysical Research Letters. 

The code may also be cited directly as:

> Clements, T. (2023) GRAPES.jl – GRAph Predcition of Earthquake Shaking in Julia (Version 1.0.0), U.S. Geological Survey Software Release, https://doi.org/10.5066/P97FBHTL.

GRAPES.jl is IPDS Record Number IP-159460. 



