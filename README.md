# ReCo

We reproduce ReCo, a contrastive learning framework designed at a regional level
to assist learning in semantic segmentation. ReCo performs pixel-level
contrastive learning on a sparse set of hard negative pixels, with minimal
additional memory footprint. ReCo is easy to implement, being built on top of
off-the-shelf segmentation networks, and consistently improves performance,
achieving more accurate segmentation boundaries and faster convergence. The
strongest effect is in semi-supervised learning with very few labels. With ReCo,
we achieve high quality semantic segmentation model, requiring only 5 examples
of each semantic class.

### Scripts

|Python script | Use case |
| :--- | :--- |
| data/cityscapes_data_loader.py | Dataloader for CityScapes |
| data/pascal_data_loader.py | Dataloader for Pascal VOC |
| utils/img_processing.py | Module for image processing |
| utils/mixing.py | Module for image data augmentation: ClassMix, CutMix |
| network/deeplabv3.py | DeepLabV3+ architecture with ResNet-101 backbone |
| network/mean_ts.py | Mean teacher framework |
| network/reconet.py | ReCo architecture with ReCo representation head |
| trainers/supervised_full.py | Supervised segmentation pipeline for Full Labels |
| trainers/semi_supervised_full.py | Semi-supervised segmentation pipeline for Full Labels |
| trainers/supervised_partial.py | Supervised segmentation pipeline for Partial Labels |
| trainers/semi_supervised_partial.py | Semi-supervised segmentation pipeline for Partial Labels |

### How to Run

1. Put the Pascal VOC and CityScapes datasets in `dataset` folder
    - `datasets/cityscapes/`: CityScapes dataset
    - `datasets/pascal/`: Pascal VOC dataset
2. Install the required packages
   ```
   pip install -r requirements.txt
   ```
3. Run the experiments
    - Full Labels Supervised Segmentation
        ```
        python -m trainers.supervised_full --dataset cityscapes --data-path dataset/cityscapes \
            --num-labeled 150 --val-interval 500 --model reconet
        ```
    - Full Labels Semi-supervised Segmentation with ClassMix
        ```
        python -m trainers.semi_supervised_full --dataset cityscapes --data-path dataset/cityscapes \
            --num-labeled 150 --val-interval 500 --model reconet --mixing-strategy classmix
        ```
    - Full Labels Semi-supervised Segmentation with ClassMix and ReCo Loss
        ```
        python -m trainers.semi_supervised_full --dataset cityscapes --data-path dataset/cityscapes \
            --num-labeled 150 --val-interval 500 --model reconet  --mixing-strategy classmix \
            --reco --reco-weight 1.0 --reco-temp 0.5 --reco-num-queries 256 --reco-num-negatives 256 --reco-threshold 0.97 
        ```
    - Partial Labels Supervised Segmentation
        ```
        python -m trainers.supervised_full --dataset cityscapes --data-path dataset/cityscapes \
            --label-ratio p1 --seed 0 --val-interval 500 --model reconet
        ```
    - Partial Labels Semi-supervised Segmentation with ClassMix
        ```
        python -m trainers.supervised_full --dataset cityscapes --data-path dataset/cityscapes \
            --label-ratio p0 --seed 0 --val-interval 500 --model reconet --mixing-strategy classmix
        ```
    - Partial Labels Semi-supervised Segmentation with ClassMix and ReCo Loss
        ```
        python -m trainers.supervised_full --dataset cityscapes --data-path dataset/cityscapes \
            --label-ratio p0 --seed 0 --val-interval 500 --model reconet --mixing-strategy classmix \
            --reco --reco-weight 1.0 --reco-temp 0.5 --reco-num-queries 256 --reco-num-negatives 256 --reco-threshold 0.97 
        ```



### Tasks

#### Enan

- [x] Implement supervised model (ResNet + DeepLabv3+)
- [x] Implement ReCo representation head
- [x] Implement mixing strategies
    - [x] ClassMix
    - [x] CutMix

#### Sanjeepan

- [x] Implement and integrate Reco Loss 
    - [x] Active Hard Key Sampling
    - [x] Active Hard Query Sampling
- [x] Implement Mean Teacher Student farmework
- [x] Implement supervised learning pipeline (full and partial)
- [x] Implement semi-supervised learning pipeline (full and partial)
- [x] Implement dataloaders for pascal and cityscapes
