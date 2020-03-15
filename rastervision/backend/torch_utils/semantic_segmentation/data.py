import logging
from os.path import join, basename
import glob

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, Dataset
from albumentations.augmentations.transforms import RandomSizedCrop

from rastervision.backend.torch_utils.data import DataBunch


log = logging.getLogger(__name__)


class ToTensor(object):
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, x, y):
        return (self.to_tensor(x), (255 * self.to_tensor(y)).squeeze().long())


class HandlerRandomSizedCrop:
    def __init__(self, **kwargs):
        self.obj = RandomSizedCrop(**kwargs)

    def __call__(self, x, y):
        # We run this before TeTensor, so x and y are still PIL.Image
        # If we run after ToTensor, need to convert tensor->ndarray
        # and back, as albumentations doesn't support tensor.
        out = self.obj(image=np.array(x), mask=np.array(y))
        return out["image"], out["mask"]


class ComposeTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.img_paths = glob.glob(join(data_dir, 'img', '*.png'))
        self.transforms = transforms

    def __getitem__(self, ind):
        img_path = self.img_paths[ind]
        label_path = join(self.data_dir, 'labels', basename(img_path))
        x = Image.open(img_path)
        y = Image.open(label_path)

        if self.transforms is not None:
            x, y = self.transforms(x, y)
        return (x, y)

    def __len__(self):
        return len(self.img_paths)


def build_databunch(data_dir, img_sz, batch_sz, class_names, augmentors):
    # set to zero to prevent "dataloader is killed by signal"
    # TODO fix this
    num_workers = 0

    train_dir = join(data_dir, 'train')
    valid_dir = join(data_dir, 'valid')

    random_sized_crop = HandlerRandomSizedCrop(
        p=1,
        min_max_height=(256, 512),
        height=256,
        width=256
    )
    augmentors_dict = {
        "RandomSizedCrop": random_sized_crop,
    }

    aug_transforms = []
    for augmentor in augmentors:
        try:
            aug_transforms.append(augmentors_dict[augmentor])
            log.info(f"Adding augmentor {augmentor}")
        except KeyError as e:
            log.warning('{0} is an unknown augmentor. Continuing without {0}. \
                Known augmentors are: {1}'
                        .format(e, list(augmentors_dict.keys())))

    aug_transforms = ComposeTransforms(aug_transforms + [ToTensor()])
    transforms = ComposeTransforms([ToTensor()])

    train_ds = SegmentationDataset(train_dir, transforms=aug_transforms)
    valid_ds = SegmentationDataset(valid_dir, transforms=transforms)

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_sz,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True)

    return DataBunch(train_ds, train_dl, valid_ds, valid_dl, class_names)
