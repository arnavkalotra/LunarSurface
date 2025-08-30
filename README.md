Ice Rover ML Navigation

Machine learning and super-resolution techniques for simulating lunar surface environments to support rover navigation and ice-mining exploration at the lunar poles.

Project Overview

This project develops tools to:

Simulate high-resolution lunar terrain for rover path planning.

Apply super-resolution (SwinIR) to enhance NASA elevation/slope datasets.

Improve traversability analysis for robotic rovers in ice-mining missions and pathfinding for a lunar settlement

Implement pathfinding algorithms (A*, HPA*, Dijkstra, RRT, Random Forest) with terrain cost functions.

Long-term goal: create a fine-tuned 3D AI environment for lunar ice-mining rovers.

 Background: Ice Detection

Based on NASA LCROSS mission:

Found water-ice spectral signatures at lunar poles.

Key absorption peaks:

2.8–3.2 µm (OH & H₂O bonds)

3000 nm (strongest H₂O absorption)

6000 nm (confirmed H₂O molecules)

Instruments: UV (260–650 nm), IR spectrometers, and mid-IR cameras to measure plume water content after impact.

📊 Dataset

Main datasets from NASA’s LOLA (Lunar Orbiter Laser Altimeter):

Elevation Map (LDEM_83S_10MPP_ADJ.TIF, 4.8 GB, 10 m/px)

Slope Map (LDSM_83S_10MPP_ADJ.TIF, 7.1 GB, 10 m/px)

Notes:

Polar orbit at 50 km with 28 Hz pulsing → ~50 m swath resolution.

Full polar coverage beyond ±86° latitude.

Coordinates: South polar stereographic (JPL DE421 reference).

 Super-Resolution Models
SRCNN

Crops & normalizes patches.

Encodes low-res → reconstructs high-res.

Early results: shallow network skipped fine-tuning features.

SwinIR

State-of-the-art transformer-based SR.

Tasks: super-resolution, denoising, artifact removal.

Uses PSNR & SSIM for evaluation.

Handles padding, normalization, patch attention windows.

Adaptation for .TIF data:

Downsample 10 m/px → 20 m/px.

Train with interference kernel sweeps.

Iteratively refine weights & handle NaNs.

Current Progress

Successful normalization + padding fixes.

Achieved PSNR > 50 dB, SSIM ~0.9995 at 8x enhancement.

Artifacts (slope exaggerations, crater over-enhancement) under investigation.

Gaussian post-processing filters reduce slope exaggeration.

Moving toward convolution interference for smoother blending.

