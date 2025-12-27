RESEARCH LETTER
10.1029/2023GL107389

Key Points:
• A deep learning network trained to
predict ground motion learned an
internal representation of the seismic
wavefield

• Individual neurons within the network
activate with the arrival of P waves, S
waves, surface waves, coda waves, and
ambient noise

• While trained on earthquakes in Japan,

the model generalizes well to
predicting ground motions for the 2019
Ridgecrest, CA earthquake

Supporting Information:

Supporting Information may be found in
the online version of this article.

Correspondence to:

T. Clements,
tclements@usgs.gov

Citation:

Clements, T., Cochran, E. S., Baltay, A.,
Minson, S. E., & Yoon, C. E. (2024).
GRAPES: Earthquake early warning by
passing seismic vectors through the
grapevine. Geophysical Research Letters,
51, e2023GL107389. https://doi.org/10.
1029/2023GL107389

Received 17 NOV 2023
Accepted 8 APR 2024

Author Contributions:

Conceptualization: T. Clements,
E. S. Cochran, A. Baltay, S. E. Minson,
C. E. Yoon
Data curation: T. Clements
Formal analysis: T. Clements
Investigation: T. Clements
Methodology: T. Clements, E. S. Cochran
Resources: E. S. Cochran
Software: T. Clements
Supervision: E. S. Cochran, A. Baltay,
S. E. Minson, C. E. Yoon
Validation: T. Clements
Visualization: T. Clements
Writing – original draft: T. Clements

© 2024. The Authors. Geophysical
Research Letters published by Wiley
Periodicals LLC on behalf of American
Geophysical Union.
This is an open access article under the
terms of the Creative Commons
Attribution License, which permits use,
distribution and reproduction in any
medium, provided the original work is
properly cited.

CLEMENTS ET AL.

GRAPES: Earthquake Early Warning by Passing Seismic
Vectors Through the Grapevine
T. Clements1

, and C. E. Yoon2

, E. S. Cochran2

, S. E. Minson1

, A. Baltay1

1U.S. Geological Survey, Earthquake Science Center, Moffett Field, CA, USA, 2U.S. Geological Survey, Earthquake
Science Center, Pasadena, CA, USA

Abstract Estimating an earthquake's magnitude and location may not be necessary to predict shaking in real
time; instead, wavefield‐based approaches predict shaking with few assumptions about the seismic source.
Here, we introduce GRAph Prediction of Earthquake Shaking (GRAPES), a deep learning model trained to
characterize and propagate earthquake shaking across a seismic network. We show that GRAPES’ internal
activations, which we call “seismic vectors”, correspond to the arrival of distinct seismic phases. GRAPES
builds upon recent deep learning models applied to earthquake early warning by allowing for continuous ground
motion prediction with seismic networks of all sizes. While trained on earthquakes recorded in Japan, we show
that GRAPES, without modification, outperforms the ShakeAlert earthquake early warning system on the 2019
M7.1 Ridgecrest, CA earthquake.

Plain Language Summary Have you ever heard something through the grapevine? It often takes you
by surprise to hear a message from someone other than the original source. You might have felt an earthquake in
a similar way: experiencing shaking (the message) at your location rather than movement along a fault (the
source). We apply grapevine‐style communication to earthquake early warning (EEW). The goal of EEW is to
warn people to prepare for earthquake shaking before damaging seismic waves arrive at their location. We build
on recent work that used deep learning and large earthquake data sets to predict earthquake shaking. We
developed a deep learning algorithm called GRAPES which predicts shaking in a manner similar to a game of
seismic telephone: when a seismic sensor detects shaking, it sends a message to its neighboring sensors, warning
them to expect shaking soon. These sensors then pass on the message to their more distant neighbors along the
grapevine. We show that the messages GRAPES learned to send between sensors are like seismic status
updates: “I'm seeing this type of seismic wave right now”. We applied GRAPES to the 2019 M7.1 Ridgecrest,
CA earthquake and it predicted shaking accurately and quickly.

1. Introduction

The competing goals of accuracy and speed in earthquake early warning (EEW) obey an uncertainty principle:
initial shaking alerts will be inherently inaccurate due to the constraint of using limited data from the start of
rupture, whereas accurate ground motion predictions, based on longer records of real‐time waveform data and
more sophisticated rupture models, arrive after they are needed (Minson et al., 2018; Wald, 2020). Traditionally,
EEW systems first estimate an earthquake's magnitude and location and then transform that information into a
prediction of future shaking (Allen & Melgar, 2019). An EEW system does not necessarily need to locate or
determine an earthquake's magnitude to predict ground shaking in real time (Hoshiba, 2013). Instead, wavefield‐
based approaches use the current seismic wavefield to predict future shaking. This type of approach redirects the
focus of ground shaking prediction from earthquake source physics to seismic wave propagation but still requires
assumptions about attenuation (Kodera et al., 2018) or scattering physics (Hoshiba & Aoki, 2015).

Deep learning (DL) is a compelling technique to solve data rich and well‐defined problems, such as EEW. DL
methods excel at prediction tasks by transforming raw input into internal representations at multiple levels of
abstraction (LeCun et al., 2015). DL has been applied extensively in EEW and real‐time seismology (Mousavi &
Beroza, 2022), with efforts divided into three general applications: (a) earthquake detection (Clements, 2023a;
Feng et al., 2022; Kim et al., 2021; Meier et al., 2019; Mousavi et al., 2020; Perol et al., 2018; Ross et al., 2018;
Saad et al., 2021; Xiao et al., 2021; Yano et al., 2021), (b) location and magnitude estimation (Perol et al., 2018;
Kriegerowski et al., 2019; van den Ende & Ampuero, 2020; Münchmeyer et al., 2021a; Zhu et al., 2021; Licciardi

1 of 10

Geophysical Research Letters

10.1029/2023GL107389

Writing – review & editing: T. Clements,
E. S. Cochran, A. Baltay, S. E. Minson,
C. E. Yoon

et al., 2022), and (c) ground motion prediction (Bloemheuvel et al., 2022; Datta et al., 2022; Fayaz & Gal-
asso, 2023; Hsu & Huang, 2021; Jozinović et al., 2020; Münchmeyer et al., 2021b; Otake et al., 2020).

Here, we introduce a DL model to continuously predict ground motions across a seismic network. We build upon
previous work using Graph Neural Networks (GNN) (van den Ende & Ampuero, 2020; Bloemheuvel et al., 2022)
and transformer models (Münchmeyer et al., 2021a, 2021b) for EEW. Our algorithm, GRAph Prediction of
Earthquake Shaking (GRAPES; Clements (2023b)), predicts future earthquake shaking using the previous 4‐s of
acceleration waveforms across a seismic network (Figure 1). At the heart of GRAPES is a GNN, which prop-
agates high‐dimensional representations of seismic waveforms, which we call “seismic vectors”, across a seismic
station graph (Figure 1c), and empirically learns the intrinsics of seismic wave propagation, attenuation, and
scattering.

2. Data and Methods

We interpret a seismic network as a graph of connected seismometers embedded in the continuous seismic
wavefield, where the graph's nodes are co‐located with seismometers, the graph's edges are connections between
nearby seismometers, and the graph's features are seismic waveforms (McBrearty & Beroza, 2023). We create
seismic station graphs by virtually connecting each seismometer to its 20 nearest neighboring seismometers and
to all seismometers within 30 km (Kodera et al., 2018). We additionally add 1 random, long‐range connection to
each station, where the probability of adding a long‐range connection at a distance R away scales as P(R) = R− 2
(Kleinberg, 2000). Such “small‐world” shortcuts increases signal propagation distance across a seismic station
graph (Figure S1 in Supporting Information S1) (Watts & Strogatz, 1998).

We characterize EEW as a spatio‐temporal graph learning problem. We create a data set of seismic station graphs
from 3,759 earthquakes with magnitudes between 3 and 9.1, recorded by the K‐NET and KiK‐net networks (Aoi
et al., 2011) in Japan from 1997 to 2018 (Figures S2–S4 in Supporting Information S1). Our data set is a subset of
Münchmeyer et al. (2021a)'s data set: we selected earthquakes that had at least 30 stations recording a P‐wave
arrival and only used surface stations. For each earthquake, we created overlapping seismic station graphs
starting with the P‐wave arrival at nearby stations and ending 5.0 s after the surface waves arrived at the furthest
stations. We chose random subsets of 30–100 stations for each timestep and used a 4‐s input window to balance
computational efficiency and capture longer‐period ground motion (text S1–S3 in Supporting Information S1).
The use of variable subsets of stations for each time step makes the approach flexible to station distributions and
station availability. This resulted in 59,991 unique seismic station graphs.

We create an additional 14,992, or 20% of total, station graphs that contain solely ambient noise data with the
intention that training on ambient noise would allow GRAPES to predict ground motion continuously in a
production setting. For our set of noise station graphs, we sample 4‐s windows of ambient noise recorded before
the P‐wave arrival from earthquakes rejected by our selection criteria above (Text S4 in Supporting Informa-
tion S1). We split the combined earthquake plus noise station graph data set into a training, validation, and test set
by time to mimic how operational EEW systems learn and adjust after subsequent earthquakes. The training,
validation and test set spans earthquakes and noise samples from 1997 to 2014, 2014–2017, and 2017–2018 and
account for 70%, 20%, and 10% of the data set, respectively.

GRAPES predicts future Peak Ground Acceleration (PGA) across a seismic network using continuous (i.e.,
updated each second) seismic station graphs as input. GRAPES contains 4 sequential neural networks: a con-
volutional neural network (CNN) for extracting time‐frequency features from input waveforms, a dense neural
network (DNN) for encoding time‐frequency and amplitude features into a “seismic vector”, a GNN for virtually
propagating seismic vectors across a seismic station graph, and a final DNN for decoding aggregated seismic
vectors into a future PGA prediction at each station.

At first order, the CNN distinguishes if the normalized waveform input at a particular station is a P wave, S wave
or other type of wave, for example, ambient noise, P‐wave coda or surface wave (Movie S1). The DNN then
log10 (peak amplitude) at each station (Münchmeyer
combines this phase information with the current
et al., 2021a) into a 128‐dimensional “seismic vector”.

GRAPES uses a set of 5 graph convolution layers in its GNN to propagate station‐level seismic vectors across the
seismic station graph (Z. Wu et al., 2020). These graph layers work like a game of seismic telephone: survey your
neighbors' seismic vectors, take the features with the highest activations, and pass those features to your neighbors

CLEMENTS ET AL.

2 of 10

 19448007, 2024, 9, Downloaded from https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL107389, Wiley Online Library on [12/10/2025]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons LicenseGeophysical Research Letters

10.1029/2023GL107389

Figure 1. The Graph Prediction of Earthquake Shaking (GRAPES) model (a) GRAPES takes in acceleration waveforms across a seismic network in real time.
(b) Station‐level network extracts seismic features vectors from three‐component waveforms. (c) Graph neural network propagates seismic vectors across seismic
station graph. Features are shared within k = 5 hop neighborhood. (d) Future shaking predicted in real time at each station using deep neural network.

and neighbors' neighbors (Text S5 in Supporting Information S1). As shown in Figure 1c, GRAPES's GNN passes
seismic vectors from a local to regional scale.

GRAPES predicts ground motion using a DNN, as opposed to traditional Ground Motion Models (GMMs), which
use a set of task‐specific features, such as distance from the fault and earthquake magnitude, to predict earthquake
shaking (Joyner & Boore, 1981). GRAPES’ DNN has an easier task than most GMMs: predict future shaking at
each station using a combination of amplitude and phase features at nearby stations (Figure 1d; Figure S5 in
Supporting Information S1).

We train GRAPES in a supervised fashion: given the current state of the seismic station graph, the model predicts
the logarithm of the PGA at each station over the next 40 s. We train GRAPES end‐to‐end to minimize the mean
squared error between recorded and predicted log10 acceleration across the network (Text S6; Figure S6 in
Supporting Information S1). End‐to‐end training forces each parameter of the network to optimize for predicting
ground motion rather than intermediate tasks such as determining an earthquake's magnitude or location as in
traditional EEW systems. We evaluate GRAPES's predictive performance using the real‐time seismic intensity
measure (Ir), which approximate the Japanese Meteorological Agency's (JMA) seismic intensity (IJMA) (Kodera
et al., 2018).

3. Results

While we trained GRAPES solely to predict ground motion, GRAPES's station‐level neural network (Figure 1b)
developed an internal vector space of seismic embeddings with a nuanced relationship to the seismic wavefield.
Ambient noise, body waves (P waves and S waves), the P‐wave coda, surface waves and later arriving coda waves
activate similar sets of features across distinct earthquakes and recording locations (Figures 2a–2c, Figures S7–S9
in Supporting Information S1). Some features are purely phase‐based (Figure 2b), whereas others solely encode
amplitude (Figure 2c). About half of the 128 features in GRAPES's seismic vector space activate on either pre‐
event ambient noise, P‐wave coda, or surface wave coda, rather than on direct P or S waves. We infer that
distinguishing noise from coda is important for GRAPES to intuit whether peak ground motion has already
occurred at a particular location.

We illustrate the tight linkage between input waveforms, seismic vectors, and shaking predictions with a M6.1
earthquake from the test set (Figure 3). The P wave arrives 3 s after origin time at the closest station to the
epicenter (SMN006, 8 km from epicenter; Figures 2a–2d; Figure 3). Within a half a second of the P‐wave arrival,

CLEMENTS ET AL.

3 of 10

 19448007, 2024, 9, Downloaded from https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL107389, Wiley Online Library on [12/10/2025]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons LicenseGeophysical Research Letters

10.1029/2023GL107389

Figure 2. Internal representations of the seismic wavefield. (A–C) Activations of station‐level network's features 1, 12, and 76, respectively, on 9 April 2018 M6.1
earthquake in Shimane/Hiroshima Prefectures, Japan (JMA event ID 2018040901323081; NIED (2019)). Seismograms are plotted with increasing epicentral distance
and scaled to unit amplitude. Feature activations in (A–D) and (F–I) are scaled between 0 and 1. (D) East‐west acceleration waveform and station‐level feature vector
representation at nearest station (8 km) from earthquake. Peak ground acceleration (PGA) is 504 cm/s2. Dashed vertical lines indicate arrival time of P‐wave, S‐wave,
and PGA, respectively. (E) GRAPES’ predicted future (purple), observed real‐time (solid gray), and peak observed (dashed horizontal gray; Imax) intensities 8 km from
epicenter. (F–H) Activation of graph‐level network's features 2, 46, and 27, respectively, on same earthquake. Dashed gray vertical line indicates time of P‐wave arrival
at station closest to the epicenter. (I) East‐west acceleration waveform and graph‐level feature vector representation of regional wavefield at station 66 km from same
earthquake as in (D). Dashed vertical lines indicate arrival time of P‐wave, S‐wave, and PGA at stations 8 and 66 km, respectively, from the epicenter. (J) GRAPES’
predicted future (purple), observed real‐time (gray), and peak observed (dashed gray; Imax) intensities 66 km from the epicenter.

SMN006's seismic vector switches from an ambient noise to earthquake state. Even though GRAPES's intensity
prediction (Ipred; Figure 2e, purple curve) quickly predicts the peak intensity (Imax; Figure 2e, dashed horizontal
line), if GRAPES had been operating as an EEW system during this earthquake, SMN006 would have received
little to no advance warning, as Ipred goes above Ir 5, the intensity level at which JMA provides alerts (Hoshiba
et al., 2008), only two seconds before that threshold is exceeded (Figure 3). This exemplifies the limitation of
EEW for shallow crustal earthquakes: areas of strongest shaking may not receive a warning before strong shaking
occurs (Minson et al., 2018; Wald, 2020). At station HRS008 (66 km from epicenter, Ir 3.7), the warning time
exceeds 10 s (Figure 2j).

We test GRAPES's timeliness and accuracy on a test set of 641 mostly crustal earthquakes recorded in Japan from
2017 to 2018 (NIED, 2019). The average magnitude of the test set is M4.3, while the largest event is the 2018
M6.7 Hokkaido Eastern Iburi earthquake (Figure S3 in Supporting Information S1). We evaluate GRAPES's
performance using the warning time framework of Meier et al. (2020) using a warning threshold of Ir 3.0, which is
roughly equivalent to Modified Mercalli Intensity (MMI) 4.5. We calculate a maximum possible warning time for
each station as the time between when GRAPES predicts shaking above the threshold and ground motion above
that threshold occurs. We include 1 s of processing time in our warning time estimates but do not account for alert
delivery times (McBride et al., 2023). We compare GRAPES's performance to the Propagation of Local Un-
damped Motion (PLUM) algorithm (Kodera et al., 2018), a ground motion‐based EEW algorithm currently in use
in Japan (Kodera et al., 2020), and to a hypothetical and unrealistic (Minson et al., 2018) “perfect” algorithm,
which exactly predicts shaking at all locations at the instant rupture begins (Figures 4a–4c). We use the perfect
algorithm as an overly optimistic estimate of the maximum available warning time, which for the test set falls
between the P‐wave and surface wave travel times. For instance, within 20 km from the hypocenter, there is a
minimum of 4 and maximum of 10 s of hypothetically possible warning time (Figure 4a). For Ir 3.0 shaking, the
perfect algorithm warns 50% of locations at least 19 s before shaking arrives, whereas GRAPES and PLUM warn
50% of locations with at least 7 and 3 s of warning, respectively (inset Figure 4a). Within a single earthquake,
though, GRAPES's percent of maximum possible warning increases with hypocentral distance out to 100 km
(Figure 4d). GRAPES generally overpredicts ground motions below Ir 3, which fall below the warning threshold,

CLEMENTS ET AL.

4 of 10

 19448007, 2024, 9, Downloaded from https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL107389, Wiley Online Library on [12/10/2025]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons LicenseGeophysical Research Letters

10.1029/2023GL107389

Figure 3. Shaking predictions 3− 6 s after origin time for earthquake in test set. (a) Shaking intensity map of the 8 April 2018 M6.1 earthquake in Shimane/Hiroshima
Prefectures, Japan (JMA event ID 2018040901323081; NIED (2019)). (b) Example seismic station graph for this earthquake. Stations are circles colored by the number
of hops (connections shown by lines between stations) from the nearest station (dark blue circle) to the epicenter. An arrow points to a small‐world connection (green
circle, bottom) located only one hop away, despite the longer distance from the nearest station. (c) Intensity observations (Ir) within 100 km of epicenter. Closed circles
are locations of seismic sensors, colored by Ir. Black and blue dotted lines indicate approximate location of P‐wave and S‐wave, respectively, at each time step.
(d) GRAPES real‐time intensity predictions (Ipred). (e) GRAPES intensity prediction error. Error value at bottom right is mean prediction error at that timestep across
entire seismic network, where the prediction error at a single station is given by (Ipred(t) −
earthquake.

− Imax), where Imax is the maximum Ir at each station throughout the

CLEMENTS ET AL.

5 of 10

 19448007, 2024, 9, Downloaded from https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL107389, Wiley Online Library on [12/10/2025]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons LicenseGeophysical Research Letters

10.1029/2023GL107389

Figure 4. Comparison of warning times on test set. (a)–(c) Maximum possible warning times for hypothetical “perfect” algorithm (Minson et al., 2018), GRAPES, and
Propagation of Local Undamped Motion (PLUM) algorithm (Kodera et al., 2018) using a warning threshold of seismic intensity 3.0. Inset of (a) shows empirical
cumulative distribution functions (CDF) of warning times for seismic intensity 3.0 for each algorithm. Precision and recall values for GRAPES and PLUM alerts are
shown in (b) and (c), respectively. (d) GRAPES and PLUM's percentage of maximum warning times for two largest earthquakes in test set. (e) Comparison of GRAPES
predicted and observed intensities on the test set at station level using warning threshold of seismic intensity 3.0. Site amplification factors have been removed from
GRAPES predictions (Kodera et al., 2018). Dashed line is 1:1 line for reference.

but is unbiased at higher ground motions after site amplification factors are removed (Figure 4e, Text S7 in
Supporting Information S1). We additionally compare GRAPES and PLUM's alerting accuracy using precision
and recall (Figures 4b and 4c). For the test set, GRAPES and PLUM achieve a precision of 0.48 and 0.21 and
recall of 0.78 and 0.88, respectively, on a station‐wise basis. This suggests GRAPES is less prone to false alerts
but may miss slightly more alerts than PLUM. For smaller events (M < 6), GRAPES takes on average less than 2 s
to achieve within 1 Ir unit of prediction accuracy (Figure S10 in Supporting Information S1). This is more than
half the rupture duration (3 s) of a M6 earthquake (Trugman et al., 2019). For the 2018 M6.7 Hokkaido event,
which had an estimated rupture duration of 15 s (Kobayashi et al., 2019), GRAPES's intensity predictions grow
with the earthquake, taking 5 s to predict the shaking to within 1 Ir intensity unit.

To demonstrate that GRAPES generalizes well to different areas, we compare GRAPES, without any modi-
fications, to the U.S. Geological Survey's ShakeAlert EEW system on the 2019 M7.1 Ridgecrest, CA earth-
quake (Figure 5). GRAPES works without modification with Southern California waveform data because while
its time (4‐s waveform) and channel (3‐component) input dimensions are fixed, its graph network can accept
any station geometry. ShakeAlert, which was operational at the time, estimated a M5.5, 1.6 magnitude units
less than the true magnitude, 6.9 s after the origin time, and grew to M6.3, 0.8 units less than the true magnitude
at 22 s after the origin time. ShakeAlert's underestimate of the magnitude led to underpredicted ground motion
intensities in real time. Consequentially, ShakeAlert did not send EEW alerts to Los Angeles County, even
though the system criteria then called for an alert to be sent at a predicted shaking level of 25 cm/s2 or MMI 4
(Chung et al., 2020). In our retrospective test, which considered real‐time data transmission latency (Stubailo
et al., 2021) but not processing or alert delivery times (Text S7 in Supporting Information S1), the P wave
arrived at the first operational seismometer 4 s after origin time. GRAPES could have predicted MMI
4+ shaking in portions of LA county within 6 s of origin time and at a majority of locations in LA county
within 9 s of origin time, at which time GRAPES's average error across the seismic network was 0.1 MMI units
(Figure 5). This highlights that the model generalized well to a new region with different station distributions
than the original training set.

Our retrospective analysis suggests that GRAPES can rapidly and accurately predict peak ground motions across
a region through a basic understanding of the seismic wavefield. However, behavior during small magnitude
(M < 3) earthquakes and problematic data (e.g., noise spikes) is unknown, as these data were not included in
training and testing; thus, additional analyses may be necessary prior to implementation in real time. We trained
GRAPES to predict PGA, which is useful for human perception of shaking, but not peak ground velocity, which is
important for predicting building damage (Y.‐M. Wu et al., 2003), or peak ground displacement from Global
Navigation Satellite Systems (Goldberg et al., 2021), which does not saturate. We did not optimize GRAPES’ pre‐
processing for computational efficiency but it can predict shaking for a few thousand stations in less than a second

CLEMENTS ET AL.

6 of 10

 19448007, 2024, 9, Downloaded from https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL107389, Wiley Online Library on [12/10/2025]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons LicenseGeophysical Research Letters

10.1029/2023GL107389

Figure 5. Comparison of GRAPES and ShakeAlert (Chung et al., 2020) predictions for the 2019 M7.1 Ridgecrest, CA earthquake from t = 4, 6, 8, and 10 s after origin
time. (a) Observed Modified Mercalli intensity (MMI) near the epicenter. Closed circles are locations of seismic sensors, colored by MMI. Black and blue dotted lines
indicate approximate location of P‐wave and S‐wave, respectively, at each time step. Seismic sensors with >1 s of real‐time data transmission latency are removed from
input (Stubailo et al., 2021). (b) GRAPES MMI predictions (MMIpred). (c) ShakeAlert MMI predictions. Error value at bottom left in (b) and (c) is calculated in similar
fashion to Figure 3 using MMI.

on a graphical processing unit (Figure S11 in Supporting Information S1). Finally, GRAPES has not been tested
on any M > 8 earthquakes, but predicted Ir 5+ within 5 s of origin time on a retrospective test of the 2024 M7.6
Noto Peninsula, Japan earthquake (Figure S12 in Supporting Information S1).

4. Conclusions

We provide evidence that GRAPES, a deep learning model trained on full earthquake waveforms, from ambient
noise to coda waves, encodes seismic phases in its seismic vector space. We show how GRAPES’ seismic vector
representations are linked both spatially and temporally to shaking predictions. With a P‐wave arrival at a nearby
station, GRAPES can rapidly and accurately predict future ground motion intensities at farther stations in the
seismic network, aided by long‐range connections embedded in its graph neural network. Further, we find that
GRAPES learned to use not only P waves and S waves, but additionally P‐wave coda, surface waves, and ambient
noise to understand the seismic wavefield. GRAPES’ performance on a geographically out‐of‐distribution test
event, the 2019 Ridgecrest, CA earthquake, suggests that its seismic vector space generalized rather than
memorized the training data.

CLEMENTS ET AL.

7 of 10

 19448007, 2024, 9, Downloaded from https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL107389, Wiley Online Library on [12/10/2025]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons LicenseAcknowledgments
TC was supported by the U.S. Geological
Survey Mendenhall fellowship program.
Any use of trade, firm, or product names is
for descriptive purposes only and does not
imply endorsement by the U.S.
Government.

Geophysical Research Letters

10.1029/2023GL107389

Conflict of Interest

The authors declare no conflicts of interest relevant to this study.

Data Availability Statement

The GRAPES. jl software was used in this study (Clements, 2023b). Earthquake waveforms were downloaded
from the National Research Institute for Earth Science and Disaster Resilience (NIED, 2019) and Southern
California Earthquake Data Center (SCEDC, 2013). Earthquake locations and magnitudes were downloaded from
the Japan Meteorological Agency (JMA) catalog (https://www.data.jma.go.jp/svd/eqev/data/bulletin/hypo.
html). The figures were created using GMT (Wessel et al., 2019) and Makie. jl (Danisch & Krumbiegel, 2021)
software. Feature activations in Figure 2 were sorted using the sequencer algorithm (Baron & Ménard, 2021).

References

Allen, R. M., & Melgar, D. (2019). Earthquake early warning: Advances, scientific challenges, and societal needs. Annual Review of Earth and

Planetary Sciences, 47(1), 361–388. https://doi.org/10.1146/annurev‐earth‐053018‐060457

Aoi, S., Kunugi, T., Nakamura, H., & Fujiwara, H. (2011). Deployment of new strong motion seismographs of k‐net and kik‐net. Earthquake Data

in Engineering Seismology: Predictive Models. Data Management and Networks, 167–186.

Baron, D., & Ménard, B. (2021). Extracting the main trend in a data set: The sequencer algorithm. The Astrophysical Journal, 916(2), 91. https://

doi.org/10.3847/1538‐4357/abfc4d

Bloemheuvel, S., van den Hoogen, J., Jozinović, D., Michelini, A., & Atzmueller, M. (2022). Graph neural networks for multivariate time series
regression with application to seismic data. International Journal of Data Science and Analytics, 16(3), 1–16. https://doi.org/10.1007/s41060‐
022‐00349‐6

Chung, A. I., Meier, M.‐A., Andrews, J., Böse, M., Crowell, B. W., McGuire, J. J., & Smith, D. E. (2020). Shakealert earthquake early warning
system performance during the 2019 ridgecrest earthquake sequence. Bulletin of the Seismological Society of America, 110(4), 1904–1923.
https://doi.org/10.1785/0120200032

Clements, T. (2023a). Earthquake detection with tinyml. Seismological Research Letters, 94(4), 2030–2039. https://doi.org/10.1785/0220220322
Clements, T. (2023b). Grapes.jl ‐ Graph prediction of earthquake shaking in julia (version 1.0.0) [software]. U.S. Geological Survey. https://doi.

org/10.5066/P97FBHTL

Danisch, S., & Krumbiegel, J. (2021). Makie. jl: Flexible high‐performance data visualization for julia. Journal of Open Source Software, 6(65),

3349. https://doi.org/10.21105/joss.03349

Datta, A., Wu, D. J., Zhu, W., Cai, M., & Ellsworth, W. L. (2022). Deepshake: Shaking intensity prediction using deep spatiotemporal rnns for

earthquake early warning. Seismological Society of America, 93(3), 1636–1649. https://doi.org/10.1785/0220210141

Fayaz, J., & Galasso, C. (2023). A deep neural network framework for real‐time on‐site estimation of acceleration response spectra of seismic

ground motions. Computer‐Aided Civil and Infrastructure Engineering, 38(1), 87–103. https://doi.org/10.1111/mice.12830

Feng, T., Mohanna, S., & Meng, L. (2022). Edgephase: A deep learning model for multi‐station seismic phase picking. Geochemistry, Geophysics,

Geosystems, 23(11), e2022GC010453. https://doi.org/10.1029/2022gc010453

Goldberg, D. E., Melgar, D., Hayes, G. P., Crowell, B. W., & Sahakian, V. J. (2021). A ground‐motion model for gnss peak ground displacement.

Bulletin of the Seismological Society of America, 111(5), 2393–2407. https://doi.org/10.1785/0120210042

Hoshiba, M. (2013). Real‐time correction of frequency‐dependent site amplification factors for application to earthquake early warning. Bulletin

of the Seismological Society of America, 103(6), 3179–3188. https://doi.org/10.1785/0120130060

Hoshiba, M., & Aoki, S. (2015). Numerical shake prediction for earthquake early warning: Data assimilation, real‐time shake mapping, and
simulation of wave propagation. Bulletin of the Seismological Society of America, 105(3), 1324–1338. https://doi.org/10.1785/0120140280
Hoshiba, M., Kamigaichi, O., Saito, M., Tsukada, S., & Hamada, N. (2008). Earthquake early warning starts nationwide in Japan. EOS,

Transactions American geophysical union, 89(8), 73–74. https://doi.org/10.1029/2008eo080001

Hsu, T.‐Y., & Huang, C.‐W. (2021). Onsite early prediction of pga using cnn with multi‐scale and multi‐domain p‐waves as input. Frontiers in

Earth Science, 9, 626908. https://doi.org/10.3389/feart.2021.626908

Joyner, W. B., & Boore, D. M. (1981). Peak horizontal acceleration and velocity from strong‐motion records including records from the 1979
imperial valley, California, earthquake. Bulletin of the Seismological Society of America, 71(6), 2011–2038. https://doi.org/10.1785/
bssa0710062011

Jozinović, D., Lomax, A., Štajduhar, I., & Michelini, A. (2020). Rapid prediction of earthquake ground shaking intensity using raw waveform data

and a convolutional neural network. Geophysical Journal International, 222(2), 1379–1389. https://doi.org/10.1093/gji/ggaa233

Kim, G., Ku, B., Ahn, J.‐K., & Ko, H. (2021). Graph convolution networks for seismic events classification using raw waveform data from

multiple stations. IEEE Geoscience and Remote Sensing Letters, 19, 1–5. https://doi.org/10.1109/lgrs.2021.3127874

Kleinberg, J. M. (2000). Navigation in a small world. Nature, 406(6798), 845. https://doi.org/10.1038/35022643
Kobayashi, H., Koketsu, K., & Miyake, H. (2019). Rupture process of the 2018 hokkaido eastern iburi earthquake derived from strong motion and

geodetic data. Earth Planets and Space, 71, 1–9. https://doi.org/10.1186/s40623‐019‐1041‐7

Kodera, Y., Hayashimoto, N., Moriwaki, K., Noguchi, K., Saito, J., Akutagawa, J., et al. (2020). First‐year performance of a nationwide
earthquake early warning system using a wavefield‐based ground‐motion prediction algorithm in Japan. Seismological Research Letters,
91(2A), 826–834. https://doi.org/10.1785/0220190263

Kodera, Y., Yamada, Y., Hirano, K., Tamaribuchi, K., Adachi, S., Hayashimoto, N., et al. (2018). The propagation of local undamped motion
(plum) method: A simple and robust seismic wavefield estimation approach for earthquake early warning. Bulletin of the Seismological Society
of America, 108(2), 983–1003. https://doi.org/10.1785/0120170085

Kriegerowski, M., Petersen, G. M., Vasyura‐Bathke, H., & Ohrnberger, M. (2019). A deep convolutional neural network for localization of
clustered earthquakes based on multistation full waveforms. Seismological Research Letters, 90(2A), 510–516. https://doi.org/10.1785/
0220180320

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436–444. https://doi.org/10.1038/nature14539

CLEMENTS ET AL.

8 of 10

 19448007, 2024, 9, Downloaded from https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL107389, Wiley Online Library on [12/10/2025]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons LicenseGeophysical Research Letters

10.1029/2023GL107389

Licciardi, A., Bletery, Q., Rouet‐Leduc, B., Ampuero, J.‐P., & Juhel, K. (2022). Instantaneous tracking of earthquake growth with elastogravity

signals. Nature, 606(7913), 319–324. https://doi.org/10.1038/s41586‐022‐04672‐7

McBrearty, I. W., & Beroza, G. C. (2023). Earthquake phase association with graph neural networks. Bulletin of the Seismological Society of

America, 113(2), 524–547. https://doi.org/10.1785/0120220182

McBride, S. K., Sumy, D. F., Llenos, A. L., Parker, G. A., McGuire, J., Saunders, J. K., et al. (2023). Latency and geofence testing of wireless
emergency alerts intended for the shakealert® earthquake early warning system for the west coast of the United States of America. Safety
Science, 157, 105898. https://doi.org/10.1016/j.ssci.2022.105898

Meier, M.‐A., Kodera, Y., Böse, M., Chung, A., Hoshiba, M., Cochran, E., et al. (2020). How often can earthquake early warning systems alert
sites with high‐intensity ground motion? Journal of Geophysical Research: Solid Earth, 125(2), e2019JB017718. https://doi.org/10.1029/
2019jb017718

Meier, M.‐A., Ross, Z. E., Ramachandran, A., Balakrishna, A., Nair, S., Kundzicz, P., et al. (2019). Reliable real‐time seismic signal/noise
discrimination with machine learning. Journal of Geophysical Research: Solid Earth, 124(1), 788–800. https://doi.org/10.1029/2018jb016661
Minson, S. E., Meier, M.‐A., Baltay, A. S., Hanks, T. C., & Cochran, E. S. (2018). The limits of earthquake early warning: Timeliness of ground

motion estimates. Science Advances, 4(3), eaaq0504. https://doi.org/10.1126/sciadv.aaq0504

Mousavi, S. M., & Beroza, G. C. (2022). Deep‐learning seismology. Science, 377(6607), eabm4470. https://doi.org/10.1126/science.abm4470
Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y., & Beroza, G. C. (2020). Earthquake transformer—An attentive deep‐learning model
for simultaneous earthquake detection and phase picking. Nature Communications, 11(1), 3952. https://doi.org/10.1038/s41467‐020‐17591‐w
Münchmeyer, J., Bindi, D., Leser, U., & Tilmann, F. (2021a). Earthquake magnitude and location estimation from real time seismic waveforms

with a transformer network. Geophysical Journal International, 226(2), 1086–1104. https://doi.org/10.1093/gji/ggab139

Münchmeyer, J., Bindi, D., Leser, U., & Tilmann, F. (2021b). The transformer earthquake alerting model: A new versatile approach to earthquake

early warning. Geophysical Journal International, 225(1), 646–656. https://doi.org/10.1093/gji/ggaa609

National Research Institute for Earth Science and Disaster Resilience (NIED). (2019). Nied k‐net, kik‐net [dataset]. National Research Institute

for Earth Science and Disaster Resilience. https://doi.org/10.17598/NIED.0004

Otake, R., Kurima, J., Goto, H., & Sawada, S. (2020). Deep learning model for spatial interpolation of real‐time seismic intensity. Seismological

Society of America, 91(6), 3433–3443. https://doi.org/10.1785/0220200006

Perol, T., Gharbi, M., & Denolle, M. (2018). Convolutional neural network for earthquake detection and location. Science Advances, 4(2),

e1700578. https://doi.org/10.1126/sciadv.1700578

Ross, Z. E., Meier, M.‐A., Hauksson, E., & Heaton, T. H. (2018). Generalized seismic phase detection with deep learning. Bulletin of the

Seismological Society of America, 108(5A), 2894–2901. https://doi.org/10.1785/0120180080

Saad, O. M., Huang, G., Chen, Y., Savvaidis, A., Fomel, S., Pham, N., & Chen, Y. (2021). Scalodeep: A highly generalized deep learning
framework for real‐time earthquake detection. Journal of Geophysical Research: Solid Earth, 126(4), e2020JB021473. https://doi.org/10.1029/
2020jb021473

SCEDC. (2013). Southern California earthquake data center [dataset]. CalTech. https://doi.org/10.7909/C3WD3xH1
Stubailo, I., Alvarez, M., Biasi, G., Bhadha, R., & Hauksson, E. (2021). Latency of waveform data delivery from the southern California seismic
network during the 2019 ridgecrest earthquake sequence and its effect on shakealert. Seismological Research Letters, 92(1), 170–186. https://
doi.org/10.1785/0220200211

Trugman, D. T., Page, M. T., Minson, S. E., & Cochran, E. S. (2019). Peak ground displacement saturates exactly when expected: Implications for

earthquake early warning. Journal of Geophysical Research: Solid Earth, 124(5), 4642–4653. https://doi.org/10.1029/2018jb017093

van den Ende, M. P., & Ampuero, J.‐P. (2020). Automated seismic source characterization using deep graph neural networks. Geophysical

Research Letters, 47(17), e2020GL088690. https://doi.org/10.1029/2020gl088690

Wald, D. J. (2020). Practical limitations of earthquake early warning. Earthquake Spectra, 36(3), 1412–1447. https://doi.org/10.1177/

8755293020911388

Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of ‘small‐world’networks. Nature, 393(6684), 440–442. https://doi.org/10.1038/30918
Wessel, P., Luis, J., Uieda, L., Scharroo, R., Wobbe, F., Smith, W. H., & Tian, D. (2019). The generic mapping tools version 6. Geochemistry,

Geophysics, Geosystems, 20(11), 5556–5564. https://doi.org/10.1029/2019gc008515

Wu, Y.‐M., Teng, T.‐l., Shin, T.‐C., & Hsiao, N.‐C. (2003). Relationship between peak ground acceleration, peak ground velocity, and intensity in

taiwan. Bulletin of the Seismological Society of America, 93(1), 386–396. https://doi.org/10.1785/0120020097

Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Philip, S. Y. (2020). A comprehensive survey on graph neural networks. IEEE Transactions on

Neural Networks and Learning Systems, 32(1), 4–24. https://doi.org/10.1109/tnnls.2020.2978386

Xiao, Z., Wang, J., Liu, C., Li, J., Zhao, L., & Yao, Z. (2021). Siamese earthquake transformer: A pair‐input deep‐learning model for earthquake
detection and phase picking on a seismic array. Journal of Geophysical Research: Solid Earth, 126(5), e2020JB021444. https://doi.org/10.
1029/2020jb021444

Yano, K., Shiina, T., Kurata, S., Kato, A., Komaki, F., Sakai, S., & Hirata, N. (2021). Graph‐partitioning based convolutional neural network for
earthquake detection using a seismic array. Journal of Geophysical Research: Solid Earth, 126(5), e2020JB020269. https://doi.org/10.1029/
2020jb020269

Zhu, J., Li, S., Song, J., & Wang, Y. (2021). Magnitude estimation for earthquake early warning using a deep convolutional neural network.

Frontiers in Earth Science, 9, 653226. https://doi.org/10.3389/feart.2021.653226

References From the Supporting Information

Abrahamson, N. A., Silva, W. J., & Kamai, R. (2014). Summary of the ask14 ground motion relation for active crustal regions. Earthquake

Spectra, 30(3), 1025–1055. https://doi.org/10.1193/070913eqs198m

Aoi, S., Kunugi, T., & Fujiwara, H. (2004). Strong‐motion seismograph network operated by nied: K‐Net and kik‐net. Journal of Japan asso-

ciation for earthquake engineering, 4(3), 65–74. https://doi.org/10.5610/jaee.4.3_65

Bensen, G., Ritzwoller, M., Barmin, M., Levshin, A. L., Lin, F., Moschetti, M., et al. (2007). Processing seismic ambient noise data to obtain
reliable broad‐band surface wave dispersion measurements. Geophysical Journal International, 169(3), 1239–1260. https://doi.org/10.1111/j.
1365‐246x.2007.03374.x

Bezanson, J., Edelman, A., Karpinski, S., & Shah, V. B. (2017). Julia: A fresh approach to numerical computing. SIAM Review, 59(1), 65–98.

https://doi.org/10.1137/141000671

Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., & Goyal, P. (2016). End to end learning for self‐driving cars. arXiv preprint

arXiv:1604.07316.

CLEMENTS ET AL.

9 of 10

 19448007, 2024, 9, Downloaded from https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL107389, Wiley Online Library on [12/10/2025]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons LicenseGeophysical Research Letters

10.1029/2023GL107389

Boore, D. M., Stewart, J. P., Seyhan, E., & Atkinson, G. M. (2014). Nga‐west2 equations for predicting pga, pgv, and 5% damped psa for shallow

crustal earthquakes. Earthquake Spectra, 30(3), 1057–1085. https://doi.org/10.1193/070113eqs184m

Campbell, K. W., & Bozorgnia, Y. (2014). Nga‐west2 ground motion model for the average horizontal components of pga, pgv, and 5% damped

linear acceleration response spectra. Earthquake Spectra, 30(3), 1087–1115. https://doi.org/10.1193/062913eqs175m

Chiou, B. S.‐J., & Youngs, R. R. (2014). Update of the chiou and youngs nga model for the average horizontal component of peak ground motion

and response spectra. Earthquake Spectra, 30(3), 1117–1153. https://doi.org/10.1193/072813eqs219m

Given, D. D., Allen, R. M., Baltay, A. S., Bodin, P., Cochran, E. S., & Creager, K. (2018). Revised technical implementation plan for the

shakealert system—An earthquake early warning system for the west coast of the United States (Tech. Rep.). US Geological Survey

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International

conference on machine learning (pp. 448–456).

Japan Meteorological Agency. (2020). Jma travel time table [dataset]. Japan Meteorological Agency. Retrieved from https://www.data.jma.go.jp/

svd/eqev/data/bulletin/catalog/appendix/trtime/trt_e.html

Kanno, T., Narita, A., Morikawa, N., Fujiwara, H., & Fukushima, Y. (2006). A new attenuation relation for strong ground motion in Japan based

on recorded data. Bulletin of the Seismological Society of America, 96(3), 879–897. https://doi.org/10.1785/0120050138

Karim, K. R., & Yamazaki, F. (2002). Correlation of jma instrumental seismic intensity with strong motion parameters. Earthquake Engineering

& Structural Dynamics, 31(5), 1191–1212. https://doi.org/10.1002/eqe.158

Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv. preprint arXiv:1412.6980.
Kunugi, T., Aoi, S., Nakamura, H., Suzuki, W., Morikawa, N., & Fujiwara, H. (2013). An improved approximating filter for real‐time calculation

of seismic intensity. Zisin, 2(60), 223–230. https://doi.org/10.4294/zisin.65.223

Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. In Proceedings of the 27th international con-

ference on machine learning (icml‐10) (pp. 807–814).

Scherer, D., Müller, A., & Behnke, S. (2010). Evaluation of pooling operations in convolutional architectures for object recognition. In Inter-

national conference on artificial neural networks (pp. 92–101).

Worden, C., Gerstenberger, M., Rhoades, D., & Wald, D. (2012). Probabilistic relationships between ground‐motion parameters and modified
mercalli intensity in California. Bulletin of the Seismological Society of America, 102(1), 204–221. https://doi.org/10.1785/0120110156

CLEMENTS ET AL.

10 of 10

 19448007, 2024, 9, Downloaded from https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL107389, Wiley Online Library on [12/10/2025]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
