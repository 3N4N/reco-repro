import torch
import torchvision.transforms.functional as TF
from . import mixing

def normalize(x):
    x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return x

def denormalize(x):
    x = TF.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    x = TF.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    return x

def tensor_to_pil(img_tensor, mask_tensor):
    img = denormalise(img_tensor.detach.cpu())
    img = TF.to_pil_image(img)

    mask = mask_tensor.float() / 255.
    mask = TF.to_pil_image(mask.unsqueeze(0).detach().cpu())

    return im, mask


def batch_transform(dataset, imgs, masks, do_scale, do_randcrop, do_augmentation):
    img_list, mask_list = [], []
    device = imgs.device

    for k in range(imgs.shape[0]):
        img_pil = denormalize(imgs[k].detach().cpu())
        img_pil = TF.to_pil_image(img_pil)

        mask_pil = masks[k].float() / 255.
        mask_pil = TF.to_pil_image(mask_pil.detach().cpu())

        aug_img, aug_mask = dataset.apply_transformations(
            img_pil, mask_pil, do_scale, do_randcrop, do_augmentation
        )
        img_list.append(aug_img.unsqueeze(0))
        mask_list.append(aug_mask.unsqueeze(0))

    aug_imgs, aug_masks = torch.cat(img_list).to(device), torch.cat(mask_list).to(device)
    return aug_imgs, aug_masks

def augment_unlabeled_batch(dataset, unlabeled_imgs, pseudo_labels, mixing_strategy):
    aug_imgs, aug_labels = batch_transform(
        dataset, unlabeled_imgs, pseudo_labels,
        do_scale=True, do_randcrop=False, do_augmentation=False
    )

    aug_imgs, aug_labels = mix_batch(
        unlabeled_imgs, pseudo_labels, strategy=mixing_strategy
    )

    aug_imgs, aug_labels = batch_transform(
        dataset, aug_imgs, aug_labels,
        do_scale=False, do_randcrop=True, do_augmentation=True
    )

    return aug_imgs, aug_labels

def mix_batch(imgs, labels, strategy):
    device = imgs.device
    if strategy == 'classmix':
        mixed_imgs, mixed_labels = classmix_batch(imgs, labels)
    elif strategy == 'cutmix':
        mixed_imgs, mixed_labels = mixing.cutmix(imgs.detach(), labels.detach())
    else:
        raise TypeError("Mode must be any of these: classmix, cutmix")
    mixed_imgs, mixed_labels = mixed_imgs.to(device), mixed_labels.to(device)
    return mixed_imgs, mixed_labels

def classmix_batch(imgs, labels):
    batch_size, _, H, W = imgs.shape
    device = imgs.device

    mixed_imgs, mixed_labels = [], []
    for i in range(batch_size):
        j = (i+1) % batch_size
        mixed_img, mixed_label = mixing.classmix(imgs[i], imgs[j], labels[i], labels[j])
        mixed_imgs.append(mixed_img.unsqueeze(0))
        mixed_labels.append(mixed_label.unsqueeze(0))

    mixed_imgs_tensor, mixed_labels_tensor = torch.cat(mixed_imgs).to(device), torch.cat(mixed_labels).to(device)
    return mixed_imgs_tensor, mixed_labels_tensor
