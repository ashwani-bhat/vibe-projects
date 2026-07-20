# 🧠 A Photo in a 12KB Neural Net

Don't store the pixels — store a **function**. A tiny network learns
f(x,&nbsp;y)&nbsp;→&nbsp;color until the function *is* your photo, then the whole
image fits in ~12KB of weights. And it fights JPEG at the same file size, live.

A zero-build, static web page. Training runs in your browser via TensorFlow.js
(WebGL); your photo never leaves the page.

## Run it

```bash
cd d06-photo-in-a-neural-net
python3 -m http.server 8000
# open http://localhost:8000
```

## How to use

1. Upload a photo (center-cropped to a square) or keep the demo sunset.
2. Hit **▶ compress** and watch a blur become your photo — a minute or two of
   training is plenty. The status line shows live PSNR.
3. Compare against the **JPEG panel**: the page binary-searches JPEG quality
   until the file matches the network's byte budget, so the fight is fair.
4. Switch the render to **4×** — the network is a continuous function, so unlike
   a pixel grid it can be sampled at *any* resolution.
5. **Export the photo as weights**: a JSON file of ~5,700 quantized numbers that
   *is* the image. Import it later (or on another machine) and the photo
   reappears without a single pixel being transferred.

## How it actually works

This is an **implicit neural representation** — a SIREN
([Sitzmann et al., NeurIPS 2020](https://arxiv.org/abs/2006.09661)):

- A 2→72→72→3 MLP with **sine activations** maps normalized coordinates to RGB.
  Sines (with the w0=30 frequency trick and the matching init scheme) are the
  whole magic: they let a ~5.7K-parameter net capture sharp edges that would
  turn to mush under ReLU.
- Training samples random batches of 2,048 pixels and runs Adam on the MSE
  between predicted and true colors. At 112×112 training resolution the image
  is learned to ~35dB PSNR in a few hundred steps.
- Export quantizes each tensor to uint16 with per-tensor min/max — measured in
  the smoke test to cost less than 1dB. 5,691 params × 2 bytes ≈ **11.1KB**.
- Because the representation is a function, "resolution" is a rendering choice,
  not a property of the file. The 4× render samples the same 11KB net on a
  448×448 grid.

## Files

| file | what |
|------|------|
| `index.html` | the page |
| `siren.js` | SIREN model, pixel-batch trainer, quantized export/import, continuous renderer |
| `app.js` | UI, JPEG-budget rival, training loop |
| `test/smoke.js` | headless sanity check (`npm test`) — learns a test image (+20dB PSNR), verifies quantization costs <1dB and continuous rendering stays sane |

## Things to try

- Portraits vs landscapes: smooth gradients compress beautifully; fine texture
  (hair, foliage) shows the net's capacity limit as painterly smoothing.
- Watch *what order* detail appears in during training — low frequencies first,
  edges last. That's spectral bias, live.
- Export weights, hit ↺ reset, then import — the photo teleports back.
- Compare the failure modes at the same byte budget: JPEG fails as blocky
  artifacts, the network fails as soft wobbly brushstrokes.
