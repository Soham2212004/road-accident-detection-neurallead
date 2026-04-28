# Road Accident Detection & Alert System
### Neural Architecture Submission — NeuralLead Maker v2026.4.2.1

> **Submission for the NeuralLead Head of AI Engineering Challenge**
> Built by [Soham](https://github.com/Soham2212004) · April 2026

---

## Overview

This repository contains a neural architecture designed and built in **NeuralLead Maker v2026.4.2.1** for real-time road accident detection and emergency alert generation.

The project is inspired by the original [Road Accident Detection & Alert System](https://github.com/Soham2212004/Road-Accident-Detection-Alert-System) and reimagines its core objective — binary classification of road scenes — as a spiking neural network with temporal dynamics, making it biologically realistic and suitable for real-time edge deployment.

**Core thesis:** Accidents are not single-frame events. They are temporal patterns of unusual motion, sound, and force. The architecture reflects this by combining visual, sensor, and audio modalities, processing them through dedicated feature extraction layers, fusing them via cross-modal attention, and passing the result through a biologically-inspired spike-based decision gate before triggering a binary alert output.

---

## Architecture

```
Input_Pipeline (224×224×3)    Input_Audio (64×1×1)    Input_Analog (10×1×1)
     Generator                    Generator                  Generator
         |                            |                           |
         v                            v                           v
 Image_Feature (64×1×1)     Audio_Feature (32×1×1)    Analog_Feature (16×1×1)
  Traditional + LeakyReLU    Traditional + LeakyReLU   Traditional + LeakyReLU
         |                            |                           |
         +----------------------------+---------------------------+
                                      |
                                      v
                           Fusion_Layer (128×1×1)
                          MultiHeadAttention · 4 Heads
                                      |
                                      v
                           Decision_Gate (32×1×1)
                         LIF (Leaky Integrate-and-Fire)
                                      |
                                      v
                        Alert_Response_System (2×1×1)
                          LIF · [1,0] Accident | [0,1] Safe
```

**Five-stage pipeline:** Inputs → Feature Extraction → Attention Fusion → Temporal Gate → Alert

---

## Neuron Groups

| Group | Grid | Neuron Model | Activation | Role |
|---|---|---|---|---|
| Input_Pipeline | 224×224×3 | Generator | — | Raw RGB frame ingestion |
| Input_Audio | 64×1×1 | Generator | — | Audio FFT amplitude data |
| Input_Analog | 10×1×1 | Generator | — | Speed, G-force, brake pressure |
| Image_Feature | 64×1×1 | Traditional | LeakyReLU | Visual accident feature extraction |
| Audio_Feature | 32×1×1 | Traditional | LeakyReLU | Crash sound signature extraction |
| Analog_Feature | 16×1×1 | Traditional | LeakyReLU | Kinematic pattern extraction |
| Fusion_Layer | 128×1×1 | MultiHeadAttention | Sigmoid | Cross-modal attention fusion |
| Decision_Gate | 32×1×1 | LIF | — | Temporal evidence accumulation |
| Alert_Response_System | 2×1×1 | LIF | — | Binary alert output |

---

## Connections

| Connection | Mode | Distribution | Purpose |
|---|---|---|---|
| Input_Pipeline → Image_Feature | Full | Xavier | Visual feature learning |
| Input_Audio → Audio_Feature | Full | Xavier | Audio feature learning |
| Input_Analog → Analog_Feature | Full | Xavier | Sensor feature learning |
| Image_Feature → Fusion_Layer | Full | Xavier | Visual features into fusion |
| Audio_Feature → Fusion_Layer | Full | Xavier | Audio features into fusion |
| Analog_Feature → Fusion_Layer | Full | Xavier | Sensor features into fusion |
| Fusion_Layer → Decision_Gate | Full | Normal | Fused signal to temporal gate |
| Decision_Gate → Alert_Response_System | Full | Normal | Final alert pathway |

All connections are plastic (trainable). Full connectivity ensures no spatial blind spots — critical for a safety system where missing any accident is unacceptable.

---

## Key Design Decisions

### Why Generator neurons at inputs?
Generator neurons are NeuralLead's designated input type. They convert continuous external values (pixel intensities, FFT amplitudes, sensor readings) into spike rate signals that downstream spiking neurons can process.

### Why Traditional + LeakyReLU for feature extraction?
Feature extraction requires dense, non-linear transformations that compress high-dimensional raw inputs into compact, semantically meaningful representations. LeakyReLU avoids the dying neuron problem of standard ReLU and maintains gradient flow for negative activations.

### Why MultiHeadAttention only at Fusion?
MultiHeadAttention is the correct model at this single stage because the fusion layer is the only point where inter-modality relationships exist. With 4 heads, the network simultaneously attends to different cross-modal relationships — in low-light conditions audio and analog channels receive higher weight; at the moment of impact all three spike simultaneously.

### Why LIF at the Decision Gate?
This is the most critical design choice in the architecture. LIF neurons integrate incoming current into a membrane potential over time, only firing when the potential crosses a threshold. A single anomalous frame — pothole, sharp brake, camera glare — will not cross the threshold alone. This biological mechanism directly implements temporal confirmation and false-positive prevention.

### Why Xavier initialization for early layers?
Xavier initialization maintains signal variance through the network by calculating optimal initial weight scale based on the number of input and output connections, ensuring gradients flow effectively during training.

### Why Spiking Neural Networks over CNNs?
Traditional deep learning processes each frame independently. SNNs accumulate evidence over multiple timesteps — architecturally superior for accident detection because accidents are temporal events, not single-frame events. SNNs are also significantly more energy-efficient for edge deployment on dashcam hardware.

---

## Project Settings

| Setting | Value | Reason |
|---|---|---|
| LearningRate | 0.001000 | Standard validated default — stable convergence |
| RunDurationStepsMs | 1 | High temporal resolution for spike timing |
| SimulationSpeed | RealTime | Matches real-world dashcam frame rates |
| RandomizeDatasetRowTraining | Enabled | Prevents order-dependent learning bias |
| StopSimulationAtPercentage | 100.0 | Full dataset before evaluation |
| ApplyLearningEachSteps | 1 | Weight update every step |
| BatchAccumulation | Disabled | Single-sample updates for small dataset |
| Accelerator | NeuralLeadTorchEngine | PyTorch GPU-accelerated backend |
| Quantization | float32 | Full precision — required for safety-critical task |

---

## Dataset

| Parameter | Value |
|---|---|
| Total images | 40 (20 + 20) |
| Class 1: Accident | 20 images — label [1, 0] |
| Class 2: No Accident | 20 images — label [0, 1] |
| Input group | Input_Pipeline |
| Output group | Alert_Response_System |
| Label format | One-hot encoded |
| Row randomization | Enabled |

The dataset used in this submission consists of synthetically generated pattern images for demonstration purposes — orange/black patterns for accident class, teal/blue for safe class. A production deployment would replace this with a real-world dashcam footage dataset.

---



## Tool Used

Built entirely in **[NeuralLead Maker v2026.4.2.1](https://neurallead.com)** — no external code written. The architecture, connections, neuron models, and training configuration are all defined within the NeuralLead framework.

---

## Submitted by

**Soham** · [github.com/Soham2212004](https://github.com/Soham2212004) · April 2026

*NeuralLead Head of AI Engineering Challenge Submission*
