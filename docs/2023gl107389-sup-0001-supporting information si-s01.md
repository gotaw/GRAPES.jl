GEOPHYSICAL RESEARCH LETTERS

Supporting Information for “GRAPES: Earthquake
Early Warning by Passing Seismic Vectors Through
the Grapevine

T. Clements1, E.S. Cochran2, A. Baltay1, S.E. Minson1, and C.E. Yoon2

1U.S. Geological Survey, Earthquake Science Center, Moffett Field, CA, USA, 94035

2U.S. Geological Survey, Earthquake Science Center, Pasadena, CA, USA, 91106

Any use of trade, firm, or product names is for descriptive purposes only and does not imply

endorsement by the U.S. Government.

March 26, 2024, 9:36pm

X - 2

:

Contents of this file

1. Text S1 to S7

2. Figures S1 to S12

3. Tables S1 to S5

Additional Supporting Information (Files uploaded separately)

1. Captions for Movies S1 to S3

Text S1. Waveform Pre-Processing

We apply the following steps to the training data. We downsample all waveforms to a

common sampling rate of 100 Hz. We then remove the mean of the pre-P wave ambient

noise from the entire waveform. We apply a 1-second Tukey taper at the beginning

and 10-second Tukey taper at end of each waveform. We then apply a causal high-pass

Butterworth filter at 0.25 Hz to each waveform. We then remove the tapered sections

from each waveform. We ensure that waveforms have at least 4 seconds of ambient noise

data before the theoretical P-wave arrival, predicted using the Japanese Meteorological

Agency’s (JMA) travel time table (Japan Meteorological Agency, 2020). We finally remove

the gain from each channel and convert acceleration waveforms to units of cm/s2. We

show the processing time with number of nodes (stations) in a graph for each processing

step in Fig. S11.

March 26, 2024, 9:36pm

:

X - 3

Text S2. Start and End-time Correction

Our GRAph Prediction of Earthquake Shaking (GRAPES) (Clements, 2023) algorithm

requires continuous waveform inputs from all stations over the time period of interest.

Because our dataset contains triggered waveform data, stations nearest to the epicenter

trigger, and start recording, earlier than stations further away from the epicenter. We

devised a procedure to extend each station’s pre-P wave ambient noise data such that all

stations for each event start at the same time (Fig. S4). We first remove any spurious

amplitudes in the ambient noise portion of the waveform using a 0.5 second moving

absolute maximum filter (Bensen et al., 2007). We then prepend a reversed and flipped

copy of the pre-P wave ambient noise to each waveform. This procedure preserves the

amplitude and phase of the waveform at the splice points (see Fig. S4). We repeat the

reverse-and-flip procedure as needed until the waveform starts at the same time as the first

triggered waveform. We apply a similar procedure to the end of each waveform, such that

all waveforms end at the same time. Care was required in prepending and appending noise

to triggered waveforms, as during initial training runs GRAPES was able to differentiate

recorded (true) and generated noise.

Text S3. Seismic Station Graph Dataset

Seismic station graphs are the input to the GRAPES algorithm for both training and

inference. A seismic station graph contains four parts: a station network graph, which

encodes the connections (edges) between stations (nodes) in the seismic network in binary

form; an edge distance, which encodes the relative distance between stations; a waveform

matrix, which includes 4-seconds of 100 Hz 3-component acceleration data for each station;

March 26, 2024, 9:36pm

X - 4

:

and a peak ground acceleration (PGA) target value, which gives the log10 PGA in the

next 40 seconds at each station.

We trained previous iterations of GRAPES to include absolute station locations in

the station network graph. We abandoned this approach because the during training

the model memorized the location of stations for large earthquakes. To ameliorate this

memorization, we include inter-station distance in the seismic station graph as the relative

edge distance between nodes (stations), rather than absolute station locations. The edge

distance between each pair of connected stations is given by ratios of the ground motion

model of Kanno, Narita, Morikawa, Fujiwara, and Fukushima (2006), which gives the

log10 acceleration at a distance R from a magnitude M earthquake:

log10(A) = 0.56 ∗ M − log10(R + 0.0055 ∗ 100.5∗M ) − 0.0031 ∗ R + 0.63

(1)

To get an edge distance with values between 0 and 1, we calculate the ratio of logarithmic

acceleration amplitude for a site between a distance R and R = 0 from an earthquake of

M 7. E.g.

e(R) =

log10(A; R = R, M = 7)
log10(A; R = 0, M = 7)

=

C1 − log10(R + C2) − 0.0031 ∗ R
C3

(2)

where C1 = 4.55, C2 = 17.39, and C3 = 3.31. This approach assumes magnitude-

independent attenuation.

Text S4. Noise station graph dataset

K-net and KiK-net stations save and transmit data when triggered by ground motion

exceeding 2 gals and 0.2-0.4 gals, respectively, and stop recording when 30 seconds of

March 26, 2024, 9:36pm

:

X - 5

signal is below 2 gals or 0.1 gals for K-net and KiK-net stations, respectively. K-net and

KiK-net records include the 15 seconds of data before the triggering time (Aoi et al.,

2004). Stations triggered by P waves thus have up to 10-15 seconds of ambient noise.

We extract these pre-P wave ambient noise recordings to create our noise station graph

dataset using the following procedure (Clements, 2023):

1. Select data from ≈ 9, 000 earthquakes with fewer than 30 triggered stations that

had been rejected from earthquake station graph dataset,

2. Calculate theoretical P-wave arrival time at each station,

3. Select 4-second ambient noise windows ending at least 4 seconds before P-wave

arrival,

4. Set future PGA target value as 3.3 times the median of the 3-component waveform

envelope.

We set the prediction target for ambient noise windows to be a constant times the

median of the 3-component noise window envelope because predicting the PGA value for

the next 40 seconds would require predicting ground motion before the origin time, which

we regard as physically impossible. We empirically found the constant value (= 3.3) as

the average of the ratio of the peak to median values of the 3-component noise window

envelope across all ambient noise windows in the training dataset, e.g.

1
W

W
(cid:88)

i=1

max (envelope (Ei, Ni, Zi))
median (envelope (Ei, Ni, Zi))

≈ 3.3

(3)

March 26, 2024, 9:36pm

X - 6

:

where W is the number of total noise windows, and Ei, Ni, Zi are the ith East, North,

and Vertical ambient noise windows, respectively.

Text S5. GRAPES Model Architecture

GRAPES (Clements, 2023) contains 4 underlying neural networks in sequence: a con-

volutional neural network (CNN), a deep neural network (DNN), a graph neural network

(GNN), and a DNN.

The input to the CNN network is a 4D tensor of dimensions T × D × C × N , where

T = 400 samples in the time dimension, D = 3 directions of ground motion (East,

North, Vertical), C = 1 channel, and N is the number of stations. The maxnorm function

removes the mean from each channel along the T dimension, then divides each channel

by the maximum standard deviation across channels. This preserves relative amplitudes

between channels. The Conv (convolution) and MaxPool (Scherer et al., 2010) operations

are applied along the time dimension and expand or contract in the channel dimension.

The Rectified Linear Unit (ReLu) (Nair & Hinton, 2010) is used as an activation function

for each Conv layer. The BatchNorm (Ioffe & Szegedy, 2015) regularizes input along the

channel dimension. The flatten operation concatenates the T, D, and C dimensions into

a single dimension. We then append the log10 peak amplitude from each station for the

current window to the flattened dimension (M¨unchmeyer et al., 2021a).

The dense neural network, or encoder network, transforms the a matrix X of size 337×N

stations into a 128 × N feature matrix. The input to the encoder network is a matrix

March 26, 2024, 9:36pm

:

X - 7

X of size 337 x N stations. Dense layers are of form W X + b, where W and b are a

learnable weight and bias matrix and vector, respectively. The ReLu (Nair & Hinton,

2010) activation function is applied after each dense layer.

We use graph convolution layers (Morris et al., 2019) to propagate feature vectors

between stations. Our graph neural network updates a station’s feature vector using a

series of 5 weighted graph convolution layers of the form:

x′
i = σ

(cid:18)

(cid:19)

W1xi + W2 max
j∈Ni

eijxj + b

(4)

where xi is the current feature vector at station i, Ni is node i’s set of neighbors, x′

i is the

output feature vector, σ is the Rectified Linear Unit (ReLu) activation function (Nair &

Hinton, 2010), eij is the edge weight from node i to node j, and W1, and W2 are learnable

weight matrices. We pass the output of graph convolution into a deep neural network, the

decoder network, which predicts a scalar value giving the maximum log10(Acceleration)

over the next 40 seconds at each station. The input to the decoder network is a matrix X

of size 128 × N stations. Dense layers are of form W X + b, where W and b are a learnable

weight and bias matrix and vector, respectively. The ReLu activation function (Nair &

Hinton, 2010) is applied after the first dense layer. The identity function is applied after

the final dense layer. Details on the architectures of the CNN, encoder, graph and decoder

networks are provided in Tables S1-S4, respectively. At a high-level, GRAPES takes the

form of the following Julia language (Bezanson et al., 2017) code listing:

function GRAPES(

model::GRAPES_model,

# deep learning model w/ weights

March 26, 2024, 9:36pm

X - 8

:

g::GNNGraph,

# contains station graph

waveforms::AbstractArray,

# T x 3 x 1 x N waveform array

edges::AbstractArray,

# edge distance array

)

pga = model.getpga(waveforms)

# extract log10 peak acceleration

x = model.preprocess(waveforms)

# station-level convolution

x = vcat(x, pga)

# concatenate peak acceleration

x = model.encoder(x)

# station-level encoding

x = model.graph_conv(g, x, edges) # graph-level convolutions

x = model.decoder(x)

# future log10(pga) prediction

return x

end

Text S6. GRAPES Model Training We train GRAPES in an end-to-end fashion

(Bojarski et al., 2016) using the Adam optimizer (Kingma & Ba, 2014). We used the mean

squared error (MSE) between the true and predicted peak log acceleration, in cm/s2, over

the next 40 seconds for each station graph as a loss function. We trained for 25 epochs

(randomized model evaluations and updates through the training data), when validation

loss began to flatten. At each epoch we evaluate our loss on the validation set and save

a training checkpoint. We use a batch size of 16 seismic station graphs, which was the

largest base 2 number of graphs that fit within memory on our graphics processing unit

March 26, 2024, 9:36pm

during training. We chose our final model as the model from the epoch with the lowest

:

X - 9

validation loss. Parameters used for training are presented in Table S5.

Text S7. Evaluation

GRAPES (Clements, 2023) predicts future ground shaking in units of log10(P GA) in

cm/s2. We evaluate GRAPES’s prediction accuracy using the real-time intensity measure

(Ir) developed by Kodera et al. (2018). We calculate Ir after applying a four-step process:

1) apply a bandpass filter to time series (Kunugi et al., 2013) 2) convert to real-time seismic

intensity using

Ir = 2 ∗ log10 A + 0.94

(5)

where A is the maximum acceleration that occurs for longer than 0.3 seconds, in cm/s2,

3) take the vector norm of the three components (Karim & Yamazaki, 2002), and 4)

remove site amplification factors (Kodera et al., 2018). We note that filtering waveforms

could take more than a second with more than a few thousand stations (Fig. S11). We

evaluate in terms of intensity rather than acceleration to be less sensitive to prediction

errors for the largest shaking and allow comparison with the PLUM algorithm, which

similarly uses intensity (Kodera et al., 2018, 2020).

We additionally apply GRAPES to the 2019 M7.1 Ridgecrest, CA earthquake (Chung

et al., 2020), using acceleration waveforms from the Southern California Seismic Network

(CI) network, and to the 2024 M7.6 Noto Peninsula, Japan earthquake, using accleration

waveforms from K-NET and KiK-net (Aoi et al., 2011). We apply the same processing to

the Noto earthquake as we do the rest of the test set. We create seismic station graphs for

March 26, 2024, 9:36pm

X - 10

:

the Ridgecrest earthquake in a similar way to our test data from Japan. We apply minimal

processing to the Ridgecrest earthquake waveform data, namely: we remove the pre-event

mean from each trace, apply a Tukey taper to the beginning and end of the trace, and apply

a high-pass filter at 0.25 Hz. We do not apply the flip-and-repeat procedure, as continuous

data are available. We remove stations (CI.WRC2, CI.CCC, CI.SLA, CI.WVP2, CI.LRL,

CI.DTP, CI.MPM, CI.DAW, CI.CCA, CI.WNM) from our predictions that had real-time

data telemetry latency greater than 1 seconds (Stubailo et al., 2021). We create seismic

station graphs using a similar sliding 4-second waveform window starting at the origin

time to 25 seconds after the origin time. We use station network graphs with each station

connected to its N=20 nearest neighbors and one long-range connection. We compare

GRAPES’s predictions for the Ridgecrest earthquake to those of the USGS’s ShakeAlert

EEW system (Given et al., 2018), which was in operation during the earthquake. We use

ShakeAlert’s epicentral location, origin time and magnitude estimates for the Ridgecrest

Earthquake as reported in (Chung et al., 2020). We then predict ground motions at each

station using a point-source estimate averaged over 4 ground motion models (Abrahamson

et al., 2014; Boore et al., 2014; Campbell & Bozorgnia, 2014; Chiou & Youngs, 2014).

We convert observed and predicted accelerations to MMI values using equation 3 from

(Worden et al., 2012). Movies of GRAPES’ predictions for the 2018 M6.1 earthquake in

Shimane/Hiroshima Prefectures, Japan (NIED, 2019) and 2019 M7.1 Ridgecrest, CA are

provided in Movies S2 and S3.

Movie S1. Activations of all 336 neurons in GRAPES’ (Clements, 2023) convolutional

neural network (CNN) overlaid on moveout of April 9, 2018 M6.1 earthquake in Shi-

March 26, 2024, 9:36pm

:

X - 11

mane/Hiroshima Prefectures, Japan (JMA event ID 2018040901323081; NIED (2019)).

Vertical acceleration waveforms are plotted by epicentral distance and scaled to unit am-

plitude. Activations of individual neurons in the CNN are plotted each 0.25 seconds and

scaled between 0 and 1. Black and blue lines indicate theoretical arrival of P wave and S

wave, respectively.

Movie S2. GRAPES shaking predictions for the April 8, 2018 M6.1 earthquake in

Shimane/Hiroshima Prefectures, Japan (JMA event ID 2018040901323081; NIED (2019)).

Closed circles are locations of seismic sensors, colored by IJM A intensity. Black and blue

dotted lines indicate approximate location of P wave and S wave, respectively, at each time

step. (A) IJM A observations. (B) GRAPES real-time IJM A predictions (Ipred). Error value

at bottom right is mean prediction error at that timestep across entire seismic network,

where the prediction error at a single station is given by (Ipred–Imax), where Imax is the

maximum IJM A at each station for the earthquake.

Movie S3. GRAPES (Clements, 2023) and ShakeAlert (Chung et al., 2020) shaking

predictions July 6, 2019 M7.1 Ridgecrest, CA earthquake (California Institute of Tech-

nology and United States Geological Survey Pasadena, 1926). Closed circles are locations

of seismic sensors, colored by Modified Mercalli Intensity (MMI). Black and blue dotted

lines indicate approximate location of P wave and S wave, respectively, at each time step.

Green focal mechanism (California Institute of Technology and United States Geological

Survey Pasadena, 1926) is plotted at epicenter of Ridgecrest, CA earthquake. (A) In-

tensity observations. (B) GRAPES real-time MMI predictions (M M Ipred). Error value

at bottom right is mean prediction error at that timestep across entire seismic network,

March 26, 2024, 9:36pm

X - 12

:

where the prediction error at a single station is given by (M M Ipred–M M Imax), where

M M Imax is the maximum MMI value at each station for the earthquake. (C) ShakeAlert

real-time MMI predictions (M M Ipred) from Chung et al. (2020).

References

Abrahamson, N. A., Silva, W. J., & Kamai, R. (2014). Summary of the ask14 ground

motion relation for active crustal regions. Earthquake Spectra, 30 (3), 1025–1055.

Aoi, S., Kunugi, T., & Fujiwara, H. (2004). Strong-motion seismograph network operated

by nied: K-net and kik-net. Journal of Japan association for earthquake engineering,

4 (3), 65–74.

Aoi, S., Kunugi, T., Nakamura, H., & Fujiwara, H. (2011). Deployment of new strong mo-

tion seismographs of k-net and kik-net. Earthquake Data in Engineering Seismology:

Predictive Models, Data Management and Networks, 167–186.

Bensen, G., Ritzwoller, M., Barmin, M., Levshin, A. L., Lin, F., Moschetti, M., . . . Yang,

Y.

(2007). Processing seismic ambient noise data to obtain reliable broad-band

surface wave dispersion measurements. Geophysical journal international , 169 (3),

1239–1260.

Bezanson, J., Edelman, A., Karpinski, S., & Shah, V. B. (2017). Julia: A fresh approach

to numerical computing. SIAM review , 59 (1), 65–98.

Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., . . . others

(2016). End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316 .

Boore, D. M., Stewart, J. P., Seyhan, E., & Atkinson, G. M. (2014). Nga-west2 equa-

tions for predicting pga, pgv, and 5% damped psa for shallow crustal earthquakes.

March 26, 2024, 9:36pm

:

X - 13

Earthquake Spectra, 30 (3), 1057–1085.

California Institute of Technology and United States Geological Survey Pasadena. (1926).

Southern California Seismic Network. Retrieved from https://doi.org/10.7914/

SN/CI (Dataset)

Campbell, K. W., & Bozorgnia, Y. (2014). Nga-west2 ground motion model for the aver-

age horizontal components of pga, pgv, and 5% damped linear acceleration response

spectra. Earthquake Spectra, 30 (3), 1087–1115.

Chiou, B. S.-J., & Youngs, R. R. (2014). Update of the chiou and youngs nga model

for the average horizontal component of peak ground motion and response spectra.

Earthquake Spectra, 30 (3), 1117–1153.

Chung, A. I., Meier, M.-A., Andrews, J., B¨ose, M., Crowell, B. W., McGuire, J. J.,

& Smith, D. E. (2020). Shakealert earthquake early warning system performance

during the 2019 ridgecrest earthquake sequence. Bulletin of the Seismological Society

of America, 110 (4), 1904–1923.

Clements, T. (2023). Grapes.jl - graph prediction of earthquake shaking in julia (ver-

sion 1.0.0). U.S. Geological Survey. Retrieved from https://code.usgs.gov/

esc/grapes.jl/-/tree/1.0.0/

(Software;

last retrieved Feb. 27, 2024) doi:

10.5066/P97FBHTL

Given, D. D., Allen, R. M., Baltay, A. S., Bodin, P., Cochran, E. S., Creager, K., . . .

others (2018). Revised technical implementation plan for the shakealert system—an

earthquake early warning system for the west coast of the united states (Tech. Rep.).

US Geological Survey.

March 26, 2024, 9:36pm

X - 14

:

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training

by reducing internal covariate shift. In International conference on machine learning

(pp. 448–456).

Japan Meteorological Agency.

(2020).

Jma travel time table.

Japan Meteoro-

logical Agency. Retrieved from https://www.data.jma.go.jp/svd/eqev/data/

bulletin/catalog/appendix/trtime/trt e.html (Dataset)

Kanno, T., Narita, A., Morikawa, N., Fujiwara, H., & Fukushima, Y. (2006). A new at-

tenuation relation for strong ground motion in japan based on recorded data. Bulletin

of the Seismological Society of America, 96 (3), 879–897.

Karim, K. R., & Yamazaki, F. (2002). Correlation of jma instrumental seismic inten-

sity with strong motion parameters. Earthquake engineering & structural dynamics,

31 (5), 1191–1212.

Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv

preprint arXiv:1412.6980 .

Kodera, Y., Hayashimoto, N., Moriwaki, K., Noguchi, K., Saito, J., Akutagawa, J., . . .

others

(2020). First-year performance of a nationwide earthquake early warning

system using a wavefield-based ground-motion prediction algorithm in japan. Seis-

mological Research Letters, 91 (2A), 826–834.

Kodera, Y., Yamada, Y., Hirano, K., Tamaribuchi, K., Adachi, S., Hayashimoto, N., . . .

Hoshiba, M. (2018). The propagation of local undamped motion (plum) method: A

simple and robust seismic wavefield estimation approach for earthquake early warn-

ing. Bulletin of the Seismological Society of America, 108 (2), 983–1003.

March 26, 2024, 9:36pm

:

X - 15

Kunugi, T., Aoi, S., Nakamura, H., Suzuki, W., Morikawa, N., & Fujiwara, H. (2013).

An improved approximating filter for real-time calculation of seismic intensity. Zisin,

2 (60), 223–230.

Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen, J. E., Rattan, G., & Grohe,

M. (2019). Weisfeiler and leman go neural: Higher-order graph neural networks. In

Proceedings of the aaai conference on artificial intelligence (Vol. 33, pp. 4602–4609).

M¨unchmeyer, J., Bindi, D., Leser, U., & Tilmann, F. (2021a). Earthquake magnitude

and location estimation from real time seismic waveforms with a transformer network.

Geophysical Journal International , 226 (2), 1086–1104.

M¨unchmeyer, J., Bindi, D., Leser, U., & Tilmann, F. (2021b). The transformer earth-

quake alerting model: A new versatile approach to earthquake early warning. Geo-

physical Journal International , 225 (1), 646–656.

Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann

machines. In Proceedings of the 27th international conference on machine learning

(icml-10) (pp. 807–814).

National Research Institute for Earth Science and Disaster Resilience (NIED). (2019).

Nied k-net, kik-net. National Research Institute for Earth Science and Disaster Re-

silience. ([Dataset]) doi: 10.17598/NIED.0004

Scherer, D., M¨uller, A., & Behnke, S.

(2010). Evaluation of pooling operations in

convolutional architectures for object recognition.

In International conference on

artificial neural networks (pp. 92–101).

Stubailo, I., Alvarez, M., Biasi, G., Bhadha, R., & Hauksson, E. (2021). Latency of

March 26, 2024, 9:36pm

X - 16

:

waveform data delivery from the southern california seismic network during the 2019

ridgecrest earthquake sequence and its effect on shakealert. Seismological Research

Letters, 92 (1), 170–186.

Worden, C., Gerstenberger, M., Rhoades, D., & Wald, D.

(2012). Probabilistic re-

lationships between ground-motion parameters and modified mercalli intensity in

california. Bulletin of the Seismological Society of America, 102 (1), 204–221.

March 26, 2024, 9:36pm

:

X - 17

Figure S1. Hops required to traverse seismic network graph with N=20 nearest neigh-

bors from 1st station to detect test earthquake in Fig. 3.

(A) Without small-world

connection. (B) With small-world connection.

March 26, 2024, 9:36pm

X - 18

:

Figure S2. Spatial extent of k=5 graph convolutions using N = 20 nearest neighbors for

a station (gold circle) to the northwest of Tokyo. Filled circles indicate station locations,

with color indicating the graph convolution. Inset shows map of Japan and figure extent.

March 26, 2024, 9:36pm

:

X - 19

Figure S3. The GRAPES dataset. (A) Location of training (green), validation (purple),

and test (orange) earthquakes, scaled by maximum intensity. Inset map shows location

and shaking intensity of the April 9, 2018 M6.1 earthquake (NIED, 2019) shown in Fig. 3.

(B) JMA latitude vs depth for training, validation, and test earthquakes. (C) Distribution

of events by maximum JMA intensity (black) in the original catalog. Distribution of events

in final dataset using biased intensity sampling (blue), where events are included 2(IJM A−2)

times. (D) Frequency of earthquake by magnitude in the initial catalog (M¨unchmeyer et

al., 2021b).

March 26, 2024, 9:36pm

X - 20

:

Figure S4.

Reverse and flip procedure to extend triggered acceleration waveforms

into continuous waveforms.

(A) East component triggered waveform from JMA event

199710211955 recorded by KiK-NET station AICH04 (black) begins recording about 4.9

seconds after origin time (Aoi et al., 2011). Waveform extended back in time (red) to

start 7 seconds before origin time. (B) Zoom-in of splice point from A. (C) Instantaneous

phase of original (black) and extended triggered waveform. (D) Zoom-in of splice point

from C. (E) Spectrogram of original triggered waveform (black) from A. (F) Spectrogram

of extended triggered waveform (red) from A.

March 26, 2024, 9:36pm

:

X - 21

Figure S5.

Connection of graph features to GRAPES shaking prediction.

(A)

Activation of graph-level feature 118 for the April 8, 2018 M6.1 earthquake in Shi-

mane/Hiroshima Prefectures, Japan (JMA event ID 2018040901323081; NIED (2019)).

(B) Activation of graph-level feature 114 on same earthquake as in (A). (C) GRAPES

predictions on same earthquake as in (A). (D) Cosine distance between feature activations

118 and 114 and GRAPES shaking prediction. The cosine distance of two vectors (a, b)

is: = 0 when a and b are identical, = 1 when a and b are orthogonal, and = 2 when a and

b have equal magnitude but point in opposite directions.

March 26, 2024, 9:36pm

X - 22

:

Figure S6. Training (black) and validation (red) loss curves for GRAPES model using

mean squared error.

March 26, 2024, 9:36pm

:

X - 23

Figure S7. Activation of station-level network’s 1st neuron as show in Fig. 2A overlaid

on moveout of 4 earthquakes in test set (M6.7/IJM A 7.0 on 2018/09/06 in Ishikari De-

pression, M6.1 IJM A 6.0 on 2018/06/18 in Kyoto/Osaka border region, M5.0/IJM A 5.5 on

2017/06/20 in Hyuganada region, and M5.6/IJM A 5.5 on 2017/06/25 in Western Nagano

Prefecture, respectively; NIED (2019); Aoi et al. (2011)). Vertical acceleration wave-

forms are plotted by epicentral distance and scaled to unit amplitude. Activations of

the 1st neuron in the station-level network are plotted each 0.25 seconds and scaled to

unit amplitude. Black and blue lines indicate theoretical arrival of P wave and S wave,

respectively.

March 26, 2024, 9:36pm

X - 24

:

Figure S8. Activation of station-level network’s 12th neuron as show in Fig. 2B overlaid

on moveout of 4 earthquakes in test set (M6.7/IJM A 7.0 on 2018/09/06 in Ishikari Depres-

sion, M6.1 IJM A 6.0 on 2018/06/18 in Kyoto/Osaka border region, M5.0/IJM A 5.5 on

2017/06/20 in Hyuganada region, and M5.6/IJM A 5.5 on 2017/06/25 in Western Nagano

Prefecture, respectively; NIED (2019); Aoi et al. (2011)). Vertical acceleration wave-

forms are plotted by epicentral distance and scaled to unit amplitude. Activations of

the 12th neuron in the station-level network are plotted each 0.25 seconds and scaled to

unit amplitude. Black and blue lines indicate theoretical arrival of P wave and S wave,

respectively.

March 26, 2024, 9:36pm

:

X - 25

Figure S9. Activation of station-level network’s 76th neuron as show in Fig. 2C overlaid

on moveout of 4 earthquakes in test set (M6.7/IJM A 7.0 on 2018/09/06 in Ishikari Depres-

sion, M6.1 IJM A 6.0 on 2018/06/18 in Kyoto/Osaka border region, M5.0/IJM A 5.5 on

2017/06/20 in Hyuganada region, and M5.6/IJM A 5.5 on 2017/06/25 in Western Nagano

Prefecture, respectively; NIED (2019); Aoi et al. (2011)). Vertical acceleration wave-

forms are plotted by epicentral distance and scaled to unit amplitude. Activations of

the 76th neuron in the station-level network are plotted each 0.25 seconds and scaled to

unit amplitude. Black and blue lines indicate theoretical arrival of P wave and S wave,

respectively.

March 26, 2024, 9:36pm

X - 26

:

Figure S10.

GRAPES (Clements, 2023) average intensity prediction error through

time for earthquakes of magnitude. (A) 3 ≤ M < 4 (B) 4 ≤ M < 5 (C) 5 ≤ M < 6 (D)

6 ≤ M < 7. Green lines indicate intensity error through time for a single earthquake.

Black line is average intensity error through time for each magnitude bin.

March 26, 2024, 9:36pm

:

X - 27

Figure S11.

Scaling of GRAPES (Clements, 2023) processing times with nodes (sta-

tions) in graph using N = 23, 24, ..., 213 = 8, 192 nodes. Filtering is done using method of

Kunugi et al. (2013) on the CPU. Graph creation involves calculating nearest neighbors

and transforming seismograms into a single tensor for inference. Inference step includes

time spent transferring station graph to GPU and running the model.

March 26, 2024, 9:36pm

X - 28

:

Figure S12. Application of GRAPES (Clements, 2023) to the 2024 M7.6 Noto Penin-

sula, Japan earthquake (NIED, 2019) for times t = 5, 10, ..50 seconds after origin time.

(A) Real-time seismic intensity observed near the epicenter. Black and blue lines denote

approximate extent of P-wave and S-wave arrivals, respectively. Stations (filled circles) are

colored by real-time seismic intensity. (B) GRAPES intensity predictions near the epicen-

ter. (C) Real-time seismic intensity observed far from epicenter. Black box shows extent

of observations in (A) and (B). (D) GRAPES intensity predictions far from epicenter.

March 26, 2024, 9:36pm

Table S1. GRAPES Convolutional Neural Network (CNN)

:

X - 29

Kernel Size Channel Input Channel Output Activation Function

1

8

32

32

64

64

64

128

128

32

3,1

3,1

2,1

3,1

3,1

2,1

3,1

2,1

3,1

3,3

2,1

8

32

32

64

64

64

128

128

32

16

ReLu

ReLu

ReLu

ReLu

ReLu

ReLu

ReLu

Layer

maxnorm

Convolution

Convolution

BatchNorm

MaxPool

Convolution

Convolution

BatchNorm

MaxPool

Convolution

BatchNorm

MaxPool

Convolution

Convolution

MaxPool

Flatten

Append Log Peak Amplitude

March 26, 2024, 9:36pm

X - 30

:

Table S2. GRAPES Encoder Network

Layer

Input Size Output Size Activation Function

Dense

Dense

337

128

128

128

ReLu

ReLu

Table S3. GRAPES Graph Network

Layer

Aggregation Input Size Output Size Activation Function

WeightedGraphConvolution

max

WeightedGraphConvolution

max

WeightedGraphConvolution

max

WeightedGraphConvolution

max

WeightedGraphConvolution

max

128

128

128

128

128

128

128

128

128

128

ReLu

ReLu

ReLu

ReLu

ReLu

Table S4. GRAPES Decoder Network

Layer

Input Size Output Size Activation Function

Dense

Dense

128

128

128

1

ReLu

Identity

March 26, 2024, 9:36pm

:

X - 31

Table S5. GRAPES Training Parameters

Training Parameter Value

Learning Rate

3 × 10−4

Optimizer

Adam (Kingma & Ba, 2014); moments = (0.9, 0.999)

Batch size

16

Loss Function

Mean Squared Error (MSE)

Training Epochs

25

March 26, 2024, 9:36pm


