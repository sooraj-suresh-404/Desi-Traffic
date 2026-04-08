---
title: DesiTraffic OpenEnv
emoji: 🚦
colorFrom: yellow
colorTo: red
sdk: docker
app_file: inference.py
pinned: false
tags:
  - openenv
---

# DesiTraffic (OpenEnv Challenge)

## Environment Overview & Motivation
DesiTraffic simulates an Indian city intersection where traffic does not follow strict lanes and is highly heterogeneous. The primary goal is to train an agent to manage traffic light phases to minimize congestion and waiting times, while rigidly prioritizing emergency corridors for ambulances and VVIPs. Unlike standard grid environments, this models the chaotic clustering of two-wheelers and large variations in vehicle density.

## Spaces
### Observation Space
Typed via Pydantic `TrafficObservation` containing:
- `queue_lengths`: (Vector4) number of vehicles waiting per approach.
- `two_wheeler_density`: (Vector4) integer 0-100 indicating percentage of compact vehicles that can clear the intersection faster.
- `ambulance_approaching`: (Vector4) binary flags indicating emergency vehicles on approaches.
- `current_green_phase`: (int 0-4) the currently active phase.
- `phase_timer`: (int) seconds the phase has been active.

### Action Space
Typed via Pydantic `TrafficAction`.
- `next_phase`: (int) The target traffic light phase (0=N-S, 1=E-W, 2=Right N-S, 3=Right E-W, 4=All Red).
- `duration`: (int) Seconds to hold if necessary (in basic continuous usage).

### Reward Space
Typed via Pydantic `TrafficReward`.
- Penalties for sum of queue lengths (wait times) and massive penalties for stopping ambulances. Bonus for clearing vehicles (throughput).

## Tasks & Difficulties
- **Easy (`desi_traffic_easy`)**: Low traffic density, rare ambulances.
- **Medium (`desi_traffic_medium`)**: Typical rush hour traffic flow.
- **Hard (`desi_traffic_hard`)**: Peak gridlock risk with frequent ambulance/VVIP movements requiring sudden clearing.

## Setup Instructions
1. Install requirements: `pip install -r requirements.txt`
2. Run validation (if OpenEnv CLI is active): `openenv validate`
3. Run inference baseline:
   ```bash
   export HF_TOKEN="your_token"
   python inference.py
   ```

## Baseline Performance Scores
Baseline scores utilizing `gpt-4o-mini` tend to vary from 0.60 to 0.85 depending on stochastic ambulance appearances.
