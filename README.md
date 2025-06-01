# DKM-HybridSN: A Resource-Efficient Model for Hyperspectral Image Classification in Oil Spill Detection

This repository provides the implementation of the **DKM-HybridSN** model from the MIGARS 2025 paper.

## Overview

DKM-HybridSN introduces Differential K-Means clustering into the HybridSN architecture to reduce memory and computational requirements significantly, making it suitable for edge and real-time applications.

## Architecture

- DKMConv3d → DKMConv2d → DKMLinear
- UMAP-based dimensionality reduction (not included here)
- 84% parameter reduction compared to original HybridSN

## Structure

- `models/`: Contains model and DKM layer definitions
- `utils/`: Utility functions
- `main.py`: Inference with input
