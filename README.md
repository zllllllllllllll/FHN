# ##FHN: Fuzzy Hashing Network for Medical Image Retrieval

# Environment

```
pip install -r requirements.txt
```


# Datasets

This project uses ChestX-ray14 and ISIC2018 datasets. If you need to download, you can access [ChestX-ray14](https://github.com/richardborbely/ChestX-ray14_CNN) and [ISIC2018](https://github.com/yuanqing811/ISIC2018) to download.

Note: please put the datasets in the data folder


# Run
## a、embedding
**Extracting the image features of the dataset in loadimage.py and saving the results as database_images.pkl and test_images.pkl files under the path "FHN\demo\chestX-ray".**

## b、run
```
python FHN_ChestX-ray14.py
```

## c、retrieval
```
python demo.py
```
