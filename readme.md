# Neural Distance Field (NeDF)

## Project Description
Our goal is to train a deep learning model to simulate a distance field using only a few depth maps and camera transforms as training material. The estimated distance field can be used to render synthetic scenes or reconstruct entire objects through a process called sphere tracing.

This work is highly similar to Neural Radiance Fields (NeRF) in terms of inputs and architecture but differs significantly in its training process, output type, and use cases. Unlike existing literature, our approach supervises the model outputs in a novel way, potentially leading to improvements in simplicity and speed.

## Notebooks Overview

### `tiny_nedf.ipynb`
A minimal example showcasing how a deep learning model can learn to predict Signed Distance Function (SDF) values.

### `nedf_V1.ipynb`
Our first version of the model and training process:
- Uses the reference depth to stop rays during both training and inference.
- Known issue: Rays never stop on unseen data.
- **Details:** Refer to the accompanying paper (`nedf_paper.pdf`) for more information.

### `nedf_V2.ipynb`
The second version of the model and training process:
- Removes supervised ray stopping, allowing better potential on new data.
- Still requires improvements for convincing performance.
- **Details:** Refer to the accompanying paper (`nedf_paper.pdf`) for additional insights.

## How to Run the Notebooks

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
2. Run them
