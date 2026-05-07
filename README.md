# Particle Filter Robot Localization

A from-scratch implementation of a **particle filter** for 2D mobile robot localization.

The filter estimates a robot's position and orientation over time by fusing noisy odometry with landmark range/bearing measurements. It outperforms pure dead-reckoning by continuously correcting for motion drift using sensor data.

## Demo

![Particle filter demo](demo.gif)

Watch the particle cloud (orange) converge on the true robot path (blue) in real time as landmark observations correct for motion drift.

```bash
python vis_particles.py
```

## Algorithm

The particle filter maintains a set of weighted hypotheses (particles) representing possible robot poses `(x, y, θ)`. Each iteration:

1. **Predict** — propagate every particle through the velocity motion model, sampling noise to reflect motion uncertainty
2. **Update** — weight each particle by how well its predicted landmark measurements match the actual sensor readings
3. **Resample** — draw a new particle set with replacement, weighted by importance scores, concentrating particles near the true pose

The filter handles the asynchronous nature of the dataset (control commands and measurements at different timestamps) by merging both into a single chronological event timeline.

## File Structure

```
.
├── run.py               # Full particle filter — main entry point
├── vis_particles.py     # Animated live visualization of particles
├── partA.py             # Part A implementation (motion + measurement models combined)
├── motion_model.py      # Standalone velocity motion model
├── measurement_model.py # Standalone landmark measurement model
└── ds1/                 # University of Toronto UTIAS robot dataset
    ├── ds1_Control.dat
    ├── ds1_Groundtruth.dat
    ├── ds1_Measurement.dat
    ├── ds1_Landmark_Groundtruth.dat
    └── ds1_Barcodes.dat
```

## Setup

```bash
pip install matplotlib pandas numpy
```

## Usage

Run the full particle filter (must be executed from the project root):

```bash
python run.py
```

This will sequentially display:
1. Motion model test — short command sequence (no noise, then with noise)
2. Particle filter on the same sequence
3. Measurement model prediction test (printed to console)
4. Pure dead-reckoning on the full dataset
5. Full particle filter on the full dataset

For an animated live visualization of the particle cloud:

```bash
python vis_particles.py
```

## Results

Pure odometry (dead-reckoning) quickly diverges from the ground truth due to accumulated drift. The particle filter successfully tracks the robot's true path by correcting for this drift using landmark observations.

| Approach | Result |
|---|---|
| Dead-reckoning only | Diverges significantly from ground truth |
| Particle filter (750 particles) | Closely tracks the ground truth path |

## Key Parameters

| Parameter | Description | Best Value Found |
|---|---|---|
| `num_particles` | Number of pose hypotheses | 750 |
| `alpha (α1–6)` | Motion noise — higher = more spread in prediction step | 0.75 |
| `sigma_r, sigma_phi` | Sensor noise for range and bearing | 0.45 |

**Effect of tuning:**
- Too low noise params → overconfident filter that locks onto wrong poses and cannot recover
- Too high noise params → broad, jittery estimate that struggles to converge
- Best results come from a balanced ratio, with slightly lower sensor noise than motion noise

## Dataset

Uses the [UTIAS Multi-Robot Cooperative Localization and Mapping Dataset](http://asrl.utias.utoronto.ca/datasets/mrclam/) (Dataset 1). The environment contains 15 fixed landmarks and 6 robots; this implementation localizes a single robot while ignoring observations of the other 5.

## Reference

Thrun, Sebastian. *Probabilistic Robotics*. The MIT Press, 2005.
