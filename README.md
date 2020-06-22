# MuSe2020
Accompany code to reproduce the baselines of the International Multimodal Sentiment Analysis Challenge (MuSe 2020).
Latest results and up-to-date version: [Paper](https://arxiv.org/pdf/2004.14858.pdf)

## Abstract
**Mu**ltimodal **Se**ntiment Analysis in Real-life Media (MuSe) 2020 is 
a Challenge-based Workshop focusing on the tasks of sentiment
recognition, as well as emotion-target engagement and trustworthiness
detection by means of more comprehensively integrating the audio-visual
and language modalities. The purpose of MuSe 2020 is to bring together
communities from different disciplines; mainly, the audio-visual emotion
recognition community (signal-based), and the sentiment analysis
community (symbol-based). We present three distinct sub-challenges:
<span>MuSe-Wild</span>, which focuses on continuous emotion (arousal and
valence) prediction; <span>MuSe-Topic</span>, in which participants
recognise 10 domain-specific topics as the target of 3-class (low,
medium, high) emotions; and <span>MuSe-Trust</span>, in which the novel
aspect of trustworthiness is to be predicted. For each
sub-challenge, a competitive baseline for participants is set; namely,
on test we report for <span>MuSe-Wild</span> combined (valence and
arousal) CCC of *.2568*, for <span>MuSe-Topic</span> score (computed as
*0.34* x UAR + *0.66* x F1) of *76.78*% on the 10-class topic and
*40.64*% on the 3-class emotion prediction, and for
<span>MuSe-Trust</span> CCC of *.4359*.

## Introduction
**Mu**ltimodal **Se**ntiment Analysis in Real-life Media (MuSe) 2020 is
a novel Challenge-based Workshop in which *sentiment recognition*, as
well as *emotion-target engagement* and *trustworthiness detection* are
the main focus. MuSe aims to provide a testing bed for more extensively
exploring the fusion of the audio-visual and language modalities. The
core purpose of MuSe is to bring together communities from differing
computational disciplines; mainly, the sentiment analysis community
(symbol-based), and the audio-visual emotion recognition community
(signal-based).


## Baseline Systems

For each Sub-challenge, a series of state-of-the-art approaches have
been applied. 
For both Sub-Challenges *<span>MuSe-Wild</span>*, and
*<span>MuSe-Trust</span>*, the paradigm is continuous prediction of
emotional signals. For this, we have applied a Recurrent Neural Network
(RNN) with self-attention approach, and a deep audio-to-target
end-to-end approach. In addition to these models, we use Support Vector
Machines (SVMs), a multimodal Transformer and a fine-tuned NLP
Transformer Albert to predict the classes of *<span>MuSe-Topic</span>*.

### End-to-End Learning [here](https://github.com/lstappen/MuSe2020/tree/master/end2you)
-------------------

As our end-to-end baseline we use End2You; an
open-source toolkit for multimodal profiling by end-to-end deep
learning. For our purposes, we
utilise three modalities, namely, audio, visual, and textual. Our audio
model is inspired by a recently proposed emotion recognition
model, and is comprised of a convolution recurrent
neural network (CRNN). In particular, we use 3 convolution layers to
extract spatial features from the raw segments. Our visual information
is comprised of the <span>VGGface</span>features, where we use zero
vectors when the face is not detected in a frame. Finally, as text
features we use <span>FastText</span>, where we replicate the text
features that span across several segments. We concatenate all uni-modal
features and feed them to a one layer LSTM to capture the temporal
dynamics in the data before the final prediction.
Contact: panagiotis.tzirakis12@imperial.ac.uk

### Early Fusion LSTM-RNN with Self-Attention [here](https://github.com/lstappen/MuSe2020/tree/master/rnn_att)
-----------------------------------------

In order to address the sequential nature of the input features, we
utilise a Long Short-Term Memory (LSTM)-RNN based architecture. The
input feature sequences are input into two parallel LSTM-RNNs with
hidden state dimensionality equal to 40, to encode the two corresponding
query and value vector sequences. A self-attention sequence is
calculated by means of a query and key dot product using a sequence-wide
attention window. The attention and query sequences are then
concatenated. For the continuous-time tasks *<span>MuSe-Wild</span>* and
*<span>MuSe-Trust</span>*, the resulting hidden vector for each time
step is further encoded by a feed-forward layer that outputs a
one-dimensional prediction sequence per prediction target. For the
*<span>MuSe-Topic</span>* task, we instead apply global max-pooling, to
integrate the sequential information into one hidden state vector, which
is then input into a feed-forward layer to provide the logits. In the
former case, all the input samples are further segmented into 50
time-step sub-segments which are all used for training, whereas in the
latter we pad/crop all sequences to 500 steps.
Contact: georgios.rizos12@imperial.ac.uk

### Multimodal Transformer [here](https://github.com/lstappen/MuSe2020/tree/master/-)
----------------------

As baseline for the non-sequential predictions of
<span>MuSe-Topic</span>, we choose the Multimodal Transformer (MMT)
. By using aligned and unaligned vision, language, and
audio features for single label prediction, it outperformed
state-of-the-art methods in a more text-focused Multimodal Sentiment
Analysis setting. MMT merges multimodal timeseries using a feed-forward
fusion process consisting of multiple crossmodal Transformer units. At
the core of this network architecture are crossmodal attention modules
which fuse multimodal features by directly attending to low-level
features across all modalities. To predict topics, valence, and arousal
we always utilise 3 feature sets, either of our three (tri), or of only
two (bi) different modalities fed into the network. We noticed that
after approximately 20 epochs the network converged. The model uses 5
crossmodal attention heads and an initial learning rate of *10<sup>-3</sup>*.
Contact: lukas.stappen@informatik.uni-augsburg.de

### Fine-tuned Albert [here](https://github.com/lstappen/MuSe2020/tree/master/muse-topic_models/fine_tune_albert)
------

To reflect the current trend towards Transformer language models, such
as Bidirectional Encoder Representations from Transformers (BERT)
, we include one of the latest versions, Albert
:, as a purely text-based baseline model. The authors of
Albert proposed parameter reduction techniques, so that the total memory
consumption is lower while increasing the training speed. These models
supposedly scale better than the original BERT. The architecture is able
to achieve state-of-the-art results on several benchmarks, despite
having a relatively smaller number of parameters. For our purposes, we
found a supervised tuning on the train partition for 3 epochs and
balanced class weights to have the best effect. We applied a learning
rate of *10<sup>-5</sup>* for the adjusted Adam Optimiser and set *\epsilon* to
*10<sup>-8</sup>*. With a sequence length of *300*, the batch size has to be
limited to *12* samples to be trained with 32GB GPU memory.
Contact: lukas.stappen@informatik.uni-augsburg.de

### Support Vector Machines [here](https://github.com/lstappen/MuSe2020/tree/master/muse-topic_models/svms)
-----------------------

For the task of emotion prediction in the Sub-Challenge
<span>MuSe-Topic</span>only, we choose also to include results obtained
through the use of conventional and easily reproducible Support Vector
Machines (SVMs). These experiments employ the Scikit-learn toolkit, with
a LINEARSVR classifier. No standardisation or normalisation was applied
to any of the reported feature sets. The complexity parameter C was
always optimised from *10<sup>-5</sup>* to *1* during the development phase, and
the best value for C is reported. In contrast to our other approaches,
we retrain the model on a concatenation of the train and development
sets to predict the final test set result.
Contact: alice.baird@informatik.uni-augsburg.de

## Baseline Features


### Acoustic

*  <span>openSMILE</span>
*  <span>DeepSpectrum</span>
### Vision
*  <span>MTCNN</span>
*  <span>VGGface</span>
*  OpenFace
*  <span>Xception</span>
*  <span>GoCaR</span>
*  <span>OpenPose</span>
### Language
* <span>FastText</span>


## Challenges

### <span>MuSe-Wild</span>Sub-Challenge


In the *<span>MuSe-Wild</span>Sub-Challenge*, participants are
predicting the level of affective dimensions (arousal, and valence) in a
time-continuous manner from audio-visual recordings. Valence thereby is
strongly linked to the emotional component of the umbrella term of
sentiment analysis and is often used interchangeably. Timestamps to enable
modality alignment and fusion on word-, sentence-, and utterance-level
as well as several acoustic, visual and textual-based features are
pre-computed and provided with the challenge package. The evaluation
metric for this sub-challenge is *concordance correlation coefficient
(CCC)*, which is often used in similar challenges. CCC is a measure of reproducibility and
performance, which condenses information on both precision and accuracy,
is robust to changes in scale and location, and
its theoretical properties to other regression measures,
<span>e.g.,</span>(root) mean squared error, are well understood
. For the baseline for the
<span>MuSe-Wild</span>sub-challenge the mean of arousal and valence is
taken.

### <span>MuSe-Topic</span>Sub-challenge


In the *<span>MuSe-Topic</span>Sub-challenge*, participants are
predicting 10-classes of domain-specific (automotive, as given by the
chosen database) topics as the target of emotions. In addition,
three classes (low, medium, and high) of valence and arousal should be
predicted <span>i.e.,</span>for each topic segment, one valence and one
arousal value. These classes are the mean value of the temporally
aggregated continuous labels of <span>MuSe-Wild</span>, which were
divided into three equally sized classes (33%) for each label For this
sub-challenge, first, the weighted score combining (0.34)
Unweighted Average Recall (UAR) and (0.66) F1 (micro) measures
independently for each predictions (Valence, Arousal and Topic) are
calculated. We include both these factors to keep our evaluation
consistent with previous challenges. Second, the mean of the weighted scores for
Valence and Arousal (combined) is calculated. Third, to combine the mean
with the topic score the mean rank over all participants ((rank of
combined emotions result + rank of topic result)/2) is calculated for
the final performance assessment. In case two participants should have
the same mean rank, the one with the highest topic rank will be the
final winner. We believe that this composite measure is most
discriminative to meaningfully showcase performance improvements in
emotion and topic prediction, as it places importance on precision and
recall, in both a dataset-wide and class-specific manner.

### <span>MuSe-Trust</span>Sub-challenge


In the *<span>MuSe-Trust</span>Sub-challenge*, participants are
predicting a continuous trustworthiness signal from user-generated
audio-visual content in a sequential manner and are provided with
aligned valence and arousal annotations, which participants are
encouraged to explore, in a means of understanding the relationship
between emotional labels in depth and at large scale. The evaluation
metric for this sub-challenge is concordance correlation coefficient
(CCC).


## Evaluation
The easiest way is to generate a file with all the model predictions as described on the website and then score it against the aggregated labels. Good examples of how to prepare the format can be found in the muse-topic_models, e.g. multimodal transformer or albert (metric.py).