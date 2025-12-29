# Minimal, Educational NeRF (PyTorch, CPU-first)

This repository contains a deliberately tiny, readable implementation of a Neural Radiance Field (NeRF). It is designed for learning, tinkering, and clarity over performance. All components are explicit and kept simple; GPU and fancy tricks are optional.

## What NeRF is (short overview)
NeRF represents a 3D scene as a continuous function that maps a 3D position (and optionally a view direction) to color and density. Instead of storing geometry explicitly (like meshes), NeRF learns an MLP that, when queried along camera rays, can reproduce the images captured from different viewpoints.

At render time, for each pixel we shoot a ray into the scene, sample multiple points along that ray, run the MLP to get color and density at those points, and then composite those results using a volume rendering equation to get the final pixel color. Training optimizes the MLP so that rendered colors match the ground-truth images.

## Rays → Points → Colors (pipeline)
1) Rays
- For each pixel, generate a ray origin and direction using camera intrinsics (`fx, fy, cx, cy`) and a camera-to-world transform (`c2w`).
- TODO: Be explicit about pixel centers vs. corners; this project assumes pixel centers.

2) Sampled points
- Uniformly sample S distances t between near/far bounds and compute 3D points p(t) = o + t·d along each ray.
- Optional future work: stratified jitter, hierarchical sampling.

3) NeRF MLP
- The MLP takes either raw 3D positions (minimal version) or their positional encodings and outputs density (sigma ≥ 0) and color (rgb in [0,1]).
- Optional future work: view direction conditioning and skip connections.

4) Volume rendering
- Convert densities to alpha: alpha_i = 1 - exp(-sigma_i · delta_i).
- Compute transmittance and per-sample weights via alpha compositing.
- Output the per-ray color as a weighted sum of sample colors.

## What works (in this educational version)
- Tiny synthetic “dataset” generator: a couple of 64×64 images (checkerboards and solids) and simple circular camera poses.
- Ray generation from pinhole intrinsics and `c2w` transforms.
- Uniform point sampling along rays.
- Minimal MLP with separate density and color heads; ReLU for sigma, Sigmoid for RGB.
- Volume rendering with alpha compositing.
- CPU-only training loop that overfits a single view and saves an output image.

## What is missing (by design)
- No hierarchical sampling (importance sampling); left as a TODO.
- No view direction conditioning (for simplicity).
- No positional encoding by default in the training loop; the minimal path uses raw xyz.
- No dataset loader for real scenes; synthetic only.
- Limited logging/metrics (just basic loss/PSNR prints).

## What confused me (notes to revisit)
- Exact coordinate conventions between Blender/LLFF vs. this toy setup (right-handedness, camera forward axis). TODO: Reconcile with the original NeRF paper and common datasets.
- Pixel-center vs. top-left pixel coordinate assumptions. This repo uses pixel centers; verify when switching datasets.
- Whether to include π in positional encoding frequencies. This repo uses it; some implementations omit it.

## How to run (CPU)
1) Set up environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy imageio
```

2) Train
```bash
python -m nerf.train
# or
python nerf/train.py
```

What happens
- Prints loss/PSNR every ~25 iterations.
- Saves a rendered image `nerf_train_output.png` when done.

## Optional: positional encoding
- Implemented in `nerf/encoding.py`.
- If you enable it in the training loop: encode points before feeding them to the MLP and update `input_dim` to match the encoded dimensionality.

## Optional: try GPU
- Change `get_device(force_cpu=True)` to `force_cpu=False` in `nerf/train.py`. Make sure you have a CUDA-enabled PyTorch install.

## Repository structure
```
nerf/
  data.py       # tiny synthetic dataset + intrinsics
  rays.py       # ray generation per pixel
  encoding.py   # positional encoding (sin/cos)
  model.py      # minimal NeRF MLP
  render.py     # sampling + volume rendering
  train.py      # CPU-first training loop
  utils.py      # device, seeding, simple image I/O
  README.md     # this file
```

## TODOs / next steps
- Add stratified/hierarchical sampling to improve quality.
- Add view direction conditioning to the color branch.
- Plug in real datasets (Blender/LLFF) with known conventions.
- Improve logging (PSNR helper, image grids, maybe TensorBoard).


