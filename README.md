# CMDL
The implementation of our paper "Contrastive Modality-Disentangled Learning for Multimodal Recommendation" (under review).
## Requirements
python == 3.7.11

numpy == 1.21.5

scipy == 1.7.3

torch == 1.11.0

## Datasets
We use the Amazon review datasets for four categories: Baby, Sports, Clothing, and Electronics. The datasets can be downloaded from Google Drive: [Baby/Sports/Elec](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG). The data already contains extracted text and image features.

## Running the Code Command
```
cd src
python main.py --model CMDL --dataset baby
```
