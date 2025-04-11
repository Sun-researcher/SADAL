# Generalized Deepfake Detection via Source-Augment Domain Adversarial Learning

## Dataset

Download **FaceForensics++** from [here](https://github.com/ondyari/FaceForensics) and place it in the `data` directory.
Download **Pretrain weight** from [here] (https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth) and place it in the `preprocess` directory.

## Environment

Use the provided `Dockerfile` to set up the required environment.

## Data Preprocessing

Run the following commands to perform face cropping and data formatting:

```bash
python3 data/crop_dlib_ff.py
python3 data/crop_retina_ff.py
python3 data/FF_save.py
```

## Training

Create the output directory first:
```bash
mkdir -p output/ssrt
# then
python3 main.py --train_batch_size 10 --dataset ssrt --name c23 --fp16

