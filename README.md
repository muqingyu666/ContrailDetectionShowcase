# Contrail Detection Demo

This project demonstrates the detection of linear contrail features in satellite cloud imagery using directional line convolution kernels, based on the methodology described in:

Mannstein, H., Meyer, R., & Wendling, P. (1999). Operational detection of contrails from NOAA-AVHRR-data. *International Journal of Remote Sensing*, *20*(8), 1641–1660. https://doi.org/10.1080/014311699212650

## Overview

The code implements a multi-angle line detection algorithm that applies directional convolution kernels to identify linear features in cloud imagery. This approach is particularly effective for detecting contrails (condensation trails) left by aircraft in satellite imagery.

## Features

- Preprocessing of cloud imagery to enhance linear features
- Creation of directional line kernels at multiple angles (0°, 15°, 30°, 45°, etc.)
- Multi-angle convolution detection to identify linear features at various orientations
- Visualization of detection results with color-coding by line orientation
- Complete implementation of the linear feature detection portion of Figure 1 from Mannstein et al. (1999)

## Requirements

- numpy
- matplotlib
- scipy
- opencv-python
- scikit-image

This demo is intended for educational purposes to showcase the effectiveness of directional convolution kernels in identifying linear features in cloud imagery.
