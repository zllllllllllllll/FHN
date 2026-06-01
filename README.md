# FHN: Fuzzy Hashing Network for Medical Image Retrieval

# Environment
Run the following command to install all necessary dependencies:
```
pip install -r requirements.txt
```


# Project Structure
```
FHN-main/
├── .idea/                      # PyCharm Project configuration
├── add_noise/                  # Data augmentation and noise processing
│   ├── add_noise_Guass.py      # Add Gaussian noise to images
│   ├── add_random_noise.py     # Add random noise to images
│   └── add_salt_pepper_noise.py# Add salt and pepper noise to images
├── data/                       # Placeholder folder for datasets (ChestX-ray14 / ISIC2018)
├── demo/                       # Evaluation, inference and generated artifacts
│   ├── database_binary_ISIC2018(5).npy  # Extracted database binary hash codes
│   ├── database_label_ISIC2018(5).npy   # Database labels
│   ├── model_ISIC2018(5).pt             # Trained model weights checkpoint
│   ├── test_binary_ISIC2018(5).npy      # Extracted query/test binary hash codes
│   ├── test_label_ISIC2018(5).npy       # Query/test labels
│   └── demo.npy       # Query/test labels
├── log/                        # Training logs and performance evaluation histories
├── model/                      # Backbone networks and configurations
│   ├── vit-base-patch16-2224/  # Pre-trained Vision Transformer (ViT) local directory
│   ├── config.json             # Model configuration file
│   ├── gitattributes           # Git LFS configuration attributes
│   └── preprocessor_config.json# Image processor configurations (normalization, resizing)
├── utils/                      # Core algorithms and helper functions
│   ├── adsh_loss_center.py     # ADSH loss function & center loss calculation
│   ├── calc_hr.py              # Precision, Recall, and MAP calculation for retrieval
│   ├── data_processing.py      # Dataset loading, preprocessing, and augmentation utils
│   ├── FNN_center.py           # Fuzzy Neural Network components or centers mapping
│   ├── fuzzy_clustering.py     # Fuzzy clustering algorithms (e.g., FCM)
│   ├── subset_sampler.py       # Balanced sampler or subset data sampler
│   └── ViT_Fuzzy.py            # Vision Transformer fused with Fuzzy modules
├── loadimage.py                # Script for dataset loading and raw feature extraction
├── README.md                   # Project documentation
├── requirements.txt            # Python environment dependencies
└── ViTHashNet_ISIC2018_fuzzy_attention.py # Main script for training & validating ISIC2018
```


# Datasets

This project uses ChestX-ray14 and ISIC2018 datasets. If you need to download, you can access [ChestX-ray14](https://github.com/richardborbely/ChestX-ray14_CNN) and [ISIC2018](https://github.com/yuanqing811/ISIC2018) to download.

Note: please put the datasets in the data folder


# Run
## a、Feature Embedding & Extraction
Extracting the image features of the dataset in loadimage.py and saving the results as database_images.pkl and test_images.pkl files under the path "FHN\demo\ISIC2018(5)".

## b、Model Training
Train the Fuzzy Hashing Network by running the main pipeline.
```
python ViTHashNet_ISIC2018_fuzzy_attention_center.py
```

## c、Evaluation & Retrieval Demo
After completing the training, conduct actual medical image retrieval:
```
python demo.py
```
