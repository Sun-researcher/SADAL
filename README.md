# Generalized Deepfake Detection via Source-Augment Domain Adversarial Learning

# Face Forgery Detection Pipeline

## Dataset

Download **FaceForensics++** from [here](https://github.com/ondyari/FaceForensics).

## Environment

Use the provided `Dockerfile` to set up the required environment.

## Data Preprocessing

Run the following commands to perform face cropping and data formatting:

```bash
python3 data/crop_dlib_ff.py
python3 data/crop_retina_ff.py
python3 data/FF_save.py

## Training

Create the output directory first:
```bash
mkdir -p output/ssrt
python3 main.py --train_batch_size 10 --dataset vssrt --name c23 --fp16

