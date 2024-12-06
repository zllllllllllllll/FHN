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
Extracting the image features of the dataset in loadimage.py and saving the results as database_images.pkl and test_images.pkl files under the path "FHN\demo\chestX-ray".

## b、run
```
python FHN_ChestX-ray14.py
```

## c、retrieval
```
python demo.py
```


# About us
## Previous work

[[1] W. Ding, C. Liu, J. Huang, C. Cheng, and H. Ju, “ViTH-RFG: Vision transformer hashing with residual fuzzy generation for targeted attack in medical image retrieval,” IEEE Trans. Fuzzy Syst., vol. 32, no. 10, pp. 5571-5584, Oct. 2024.](https://ieeexplore.ieee.org/abstract/document/10360307)

[[2] W. Ding, T. Zhou, J. Huang, S. Jiang, T. Hou, and C.-T. Lin, “FMDNN: A fuzzy-guided multi-granular deep neural network for histopathological image classification,” IEEE Trans. Fuzzy Syst., val. 32, no. 8, pp. 4709-4723, Aug. 2024.](https://ieeexplore.ieee.org/abstract/document/10552048)

[[3] W. Ding, T. Hou, J. Huang, H. Ju, and S. Jiang, “Dynamic Evidence Fusion Neural Networks with Uncertainty Theory and Its Application in Brain Network Analysis,” Inf. Sci., val. 691, pp. 121622, Feb. 2025.](https://www.sciencedirect.com/science/article/pii/S0020025524015366)

[[4] Z. Deng, L. Liang, H. Yang, W. Zhang, Q. Lou, K. S. Choi, and S. Wang, “Enhanced multiview fuzzy clustering using double visible-hidden view cooperation and network LASSO constraint,” IEEE Trans. Fuzzy Syst., vol. 30, no. 11, pp. 4965-4979, Nov. 2022.](https://ieeexplore.ieee.org/abstract/document/9749939)

[[5] Z. Deng, R. Liu, P. Xu, K. S. Choi, W. Zhang, X. Tian, and S. Wang, “Multi-view clustering with the cooperation of visible and hidden views,” IEEE Trans. Knowl. Data Eng., vol. 34, no. 2, pp. 803-815, Feb. 2022.](https://ieeexplore.ieee.org/abstract/document/9051812)
