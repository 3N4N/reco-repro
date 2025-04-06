# ReCo

### Tasks

Enan

- [x] Implement supervised model (ResNet + DeepLabv3+)
- [x] Implement ReCo representation head
- [x] Implement mixing strategies
    - [x] ClassMix
    - [x] CutMix

Sanjeepan

- [x] Implement and integrate Reco Loss 
    - [x] Active Hard Key Sampling
    - [x] Active Hard Query Sampling
- [x] Implement Mean Teacher Student farmework
- [x] Implement supervised learning pipeline (full and partial)
- [x] Implement semi-supervised learning pipeline (full and partial)
- [x] Implement dataloaders for pascal and cityscapes


Notes:

- Labeled data is augmented in standard manner (Appendix A)
- Unlabled data is augmented  w/ mixing strategy
- Two steps of augmentation is inspired from ClassMix
    - Weak augmentation for labeled images (flip)
    - Strong augmentation for unlabled images (jitter etc.)
    - But ReCo augments labeled images w/ strong augmentation too
- Our data loaders doesn't respect batch-size from cmdline argument [ !!! ]
