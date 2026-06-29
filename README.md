# kvae-audio

Minimal inference package for KVAE-Audio.
Most of the code is adopted from [DAC](https://github.com/descriptinc/descript-audio-codec).
This is a VAE version with continuous latent space and enhanced diffusability.

## Install

First install PyTorch (adjust for your CUDA setup):

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Then install KVAE-Audio from the local package:

```bash
pip install -e ./kvae-audio
```

Tested with Python 3.14.

## Run inference

Place input wavs in `input_samples`. Reconstructions are written to `output_samples`.

```bash
python scripts/infer.py \
  --weights_path kvae-audio_1_0_weights.pt \
  --input_path input_samples \
  --output_path output_samples
```

## Evaluate reconstructions

Compare reference wavs in `input_samples` against reconstructions in `output_samples` (same filenames). Writes `metrics.csv` to `output_samples`.

```bash
python scripts/eval.py \
  --input_path input_samples \
  --output_path output_samples
```

## Python API

By default `encode` returns the mean latent (`mu`); pass `sample=True` for stochastic sampling. For end-to-end reconstruction, `model(waveform, sample_rate)["audio"]` is equivalent.

```python
import torch
import soundfile as sf
from kvae_audio import KVAEAudio

model = KVAEAudio.load("kvae-audio_1_0_weights.pt", map_location="cpu")
model.eval()

data, sr = sf.read("audio.wav")
waveform = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
length = waveform.shape[-1]

with torch.no_grad():
    latents, _, _, _ = model.encode(waveform, sample_rate=sr)
    audio = model.decode(latents)[..., :length]

sf.write("output.wav", audio.squeeze(0).T.numpy(), sr)
```

## Package layout

```
kvae-audio/
├── kvae_audio/
│   ├── model/
│   │   ├── kvae_audio.py
│   │   └── base.py
│   ├── metrics/loss.py
│   └── nn/layers.py
└── scripts/
    ├── infer.py
    └── eval.py
```
