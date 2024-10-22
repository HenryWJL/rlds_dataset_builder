# MagiClaw RLDS Dataset Builder
This repo is used for building TensorFlow datasets based on human demonstrations collected by [MagiClaw](https://appadvice.com/app/magiclaw/6661033548). The generated datasets satisfy the [RLDS format](https://github.com/google-research/rlds#dataset-format) and can be uploaded to the Open X-Embodiment dataset. Code in this repo is copied and modified from [kpertsch](https://github.com/kpertsch/rlds_dataset_builder).

## 🔧 Installation
Create a conda environment with all dependencies installed:
```bash
conda env create -f environment_ubuntu.yml
conda activate rlds_env
```

## 🔨 Building TensorFlow Dataset
Before building a TensorFlow dataset, make sure you have exported the episode folder from your iPhone and have placed it somewhere on your computer.
```bash
cd magiclaw_dataset
python build_dataset.py -ed <PATH_TO_FOLDER> -s train -it nearest
```
This will automatically generate a TensorFlow dataset at `~/tensorflow_datasets/magiclaw_dataset`.

## 📺 Visualizing TensorFlow Dataset
Once you have built the TensorFlow dataset, you can visualize it by running:
```bash
cd ..
python visualize_dataset.py
```

## 📤 Uploading Dataset
You can upload your own dataset to the Open X-Embodiment dataset through this [form](https://docs.google.com/forms/d/e/1FAIpQLSeYinS_Y5Bf1ufTnlROULVquD4gw6xY_wUBssfVYkHNaPp4LQ/viewform).

## 📥 Downloading Dataset from Web
We will soon release our dataset.

