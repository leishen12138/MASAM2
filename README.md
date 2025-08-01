## MASAM2: Frequency-Aware SAM2-UNet for Underwater Marine Animal Segmentation
### Abstract
Underwater marine animal segmentation is a challenging task due to unique environmental degradations such as light
scattering, color distortion, and motion blur. Although Deep
learning models have achieved progress, they often struggle with modeling long-range dependencies and handling
underwater-specific noise. Recent variants of the Segment
Anything Model (SAM) offer strong general-purpose segmentation capabilities but lack the ability to effectively process frequency-domain features that are critical for underwater images. To address this issue, we propose MASAM2, a
light-weight frequency-aware SAM2-UNet architecture that
integrates an adaptive frequency decomposition module and
a dual-supervision hybrid loss. The proposed module employs learnable high- and low-frequency filters to enhance
edge details and suppress scattering artifacts, while the hybrid
loss combines spatial structural constraints and frequencydomain guidance to improve segmentation of low-contrast
and blurred targets. Experiments on four MAS benchmarks
(MAS3K, RMAS, UFO120, and RUWI) demonstrate that
MASAM2 outperforms existing SOTA methods in terms of
mIoU and MAE with only a 0.16M increase in parameters,
highlighting the effectiveness of frequency-domain processing for robust underwater segmentation.
![Logo](https://github.com/leishen12138/MASAM2/blob/main/MASAM2.png)
