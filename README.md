# MagiClaw RLDS Dataset Builder

This repo functions to build a TensorFlow dataset satisfying the [RLDS format](https://github.com/google-research/rlds#dataset-format) based on a raw [MagiClaw](https://appadvice.com/app/magiclaw/6661033548) dataset. Code in this repo is copied and modified from [kpertsch](https://github.com/kpertsch/rlds_dataset_builder).

## 🔧 Installation
Create a conda environment with all dependencies installed:
```
conda env create -f environment_ubuntu.yml
conda activate rlds_env
```

## 🔨 Building TensorFlow Dataset
Before building the TensorFlow dataset, make sure you have exported the episode directory from your iPhone and have placed it somewhere on your computer. Now you can run the following command:
```
cd magiclaw_dataset
python build_dataset.py -ed <DIRECTORY> -s split -it nearest
```
This will automatically generate a TensorFlow dataset at `~/tensorflow_datasets/magiclaw_dataset`.

## 📤 Uploading Dataset
You can upload your own dataset to the Open X-Embodiment dataset through this [form](https://docs.google.com/forms/d/e/1FAIpQLSeYinS_Y5Bf1ufTnlROULVquD4gw6xY_wUBssfVYkHNaPp4LQ/viewform).

## 📥 Downloading Dataset from Web
We will soon release our dataset.

