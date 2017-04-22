We propose an audio enhancement technique that
increases the sampling rate of audio signals such
as speech or music using deep convolutional neural networks. Our model is trained on pairs of
low and high-quality audio examples; at test-time, it predicts missing samples within a low-resolution signal in an interpolation process similar to image super-resolution. Our method is
considerably simpler than previous approaches
and outperforms baselines on standard speech
and music benchmarks at 2×, 4×, and 6× up-scaling ratios. The method has practical applications in telephony, compression, and text-to-speech generation; it also introduces new architectures that could help scale recently proposed
generative models of audio.

## Audio Super-Resolution Samples

Below, you may find samples from our neural network-based audio super-resolution model and several baselines.

We will be adding more samples here as they become available.

### Single Speaker (4x upscaling)

We start with models trained and tested on different utterances from the same speaker. At 4x upscaling, the reproduction quality is very good, and it can be sometimes difficult to tell the reconstructions apart from the originals.

### Sample 1

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/4/sp1.1.4.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/4/sp1.1.4.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/4/sp1.1.4.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/4/sp1.1.4.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 2

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/4/sp1.2.4.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/4/sp1.2.4.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/4/sp1.2.4.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/4/sp1.2.4.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 3

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/4/sp1.3.4.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/4/sp1.3.4.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/4/sp1.3.4.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/4/sp1.3.4.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Single Speaker (6x upscaling)

At 6x upscaling, the low-resolution audio becomes very hard to comprehend, but our method can recover a significant fraction of the high frequency, at the cost of introducing a small amount of noise.

### Sample 1

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/6/sp1.1.6.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/6/sp1.1.6.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/6/sp1.1.6.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/6/sp1.1.6.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 2

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/6/sp1.2.6.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/6/sp1.2.6.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/6/sp1.2.6.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/6/sp1.2.6.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 3

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/6/sp1.3.6.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/6/sp1.3.6.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/6/sp1.3.6.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/sp1/6/sp1.3.6.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>


### MultiSpeaker (4x upscaling)

Next, we look at the more interesting setting where we train on one set of speakers, and test on a second set of speakers.
On the more interesting 4x task, our output is slightly noisier, but it also clearly outperforms the baseline.

### Sample 1

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.1.4.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.1.4.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.1.4.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.1.4.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 2

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.2.4.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.2.4.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.2.4.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.2.4.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 3

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.3.4.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.3.4.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.3.4.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.3.4.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 4

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.4.4.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.4.4.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.4.4.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.4.4.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 5

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.5.4.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.5.4.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.5.4.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/msp/4/msp.5.4.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Piano (4x upscaling)

At 4x upsampling, both the neural network and the cubic interpolation baseline are good at reconstructing the downsampled signal and perform comparably well.

### Sample 1

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/4/piano.1.4.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/4/piano.1.4.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/4/piano.1.4.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/4/piano.1.4.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 2

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/4/piano.2.4.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/4/piano.2.4.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/4/piano.2.4.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/4/piano.2.4.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 3

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/4/piano.3.4.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/4/piano.3.4.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/4/piano.3.4.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/4/piano.3.4.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Piano (6x upscaling)

At 6x upscaling, the problem becomes more difficult. Neural networks reconstruct more high frequency, but also introduce some background noise.

### Sample 1

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/6/piano.1.6.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/6/piano.1.6.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/6/piano.1.6.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/6/piano.1.6.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 2

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/6/piano.2.6.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/6/piano.2.6.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/6/piano.2.6.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/6/piano.2.6.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 3

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/6/piano.3.6.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/6/piano.3.6.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/6/piano.3.6.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/6/piano.3.6.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Piano (8x upscaling)

At 8x upscaling, the task is even more challenging; our baseline sounds dull, while the neural network recovers more detail, but also adds some distortion.

### Sample 1

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/8/piano.1.8.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/8/piano.1.8.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/8/piano.1.8.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/8/piano.1.8.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 2

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/8/piano.2.8.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/8/piano.2.8.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/8/piano.2.8.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/8/piano.2.8.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Sample 3

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/8/piano.3.8.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/8/piano.3.8.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/8/piano.3.8.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Cubic interpolation: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/piano/8/piano.3.8.sp.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

### Interesting samples

Finally, we are going to look at a few more samples that provide useful insights into how our method works.

### Sample 1 (4x upscaling)

Our first sample shows that our method can "hallucinate" new sounds when they can no longer be exactly recovered. Sometimes, this works correctly, as in some of the examples above. In other cases, it leads to some interesting failure modes, like the one below.

<div> High resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/extra/extra.1.4.hr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Low resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/extra/extra.1.4.lr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

<div> Super-resolution: <audio controls>
  <source src="https://raw.githubusercontent.com/kuleshov/audio-super-res/master/samples/extra/extra.1.4.pr.wav" type="audio/wav">
Your browser does not support the audio element.
</audio> </div>

