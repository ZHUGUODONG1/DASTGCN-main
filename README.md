
# DASTGCN-main

## 1. Title

### Spatio-Temporal Graph Neural Network for Traffic Forecasting with Diverse Key Heterogeneous Information Awareness

## 2. Framework
![image](Figure_2.png)            

## 3. Training

```bash
python train.py \
    --device cuda:0 \
    --data data/ChengDu_City \
    --adjdata data/ChengDu_City/adj_mat.pkl \
    --adjtype doubletransition \
    --batch_size 32 \
    --epochs 100 \
    --sample_ratio 0.333333 \
    --lambda_cl 0.4 \
    --CL true \
    --force true
```

## 4. paper
```bash
@article{zhu2026spatio,
  title={Spatio-temporal graph neural network for traffic forecasting with diverse key heterogeneous information awareness},
  author={Zhu, Guodong and Du, Bo and Niu, Yunyun},
  journal={Pattern Recognition},
  pages={114807},
  year={2026},
  publisher={Elsevier}
}
```




