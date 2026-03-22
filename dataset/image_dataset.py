import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
import torchvision.transforms.v2 as T
import os


def create_transforms(image_size: int, augment: bool = True):
    transform_list = [
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToImage(),  # Convert PIL → tensor (uint8)
        T.ToDtype(
            torch.float32, scale=True
        ),  # Convert to float32 and scale [0,255] → [0,1]
    ]

    if augment:
        transform_list.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
            ]
        )

    transform_list.extend([T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    return T.Compose(transform_list)


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, preload=False):
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.images = []
        self.labels = []
        self.preload = preload

        self.image_dirs = sorted(os.listdir(root))
        for idx, image_dir in enumerate(self.image_dirs):
            image_dir_path = os.path.join(root, image_dir)
            images_in_dir = sorted(os.listdir(image_dir_path))
            for image_file in images_in_dir:
                image_full_path = os.path.join(image_dir_path, image_file)
                if os.path.isfile(image_full_path):
                    self.image_paths.append(image_full_path)
                    self.labels.append(idx)
                    if self.preload:
                        self.images.append(self.load_image(image_full_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = None
        if self.preload:
            image = self.images[idx]
        else:
            image = self.load_image(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        else:
            image /= 255.0  # Scale to [0,1] if not using transforms
        return image, self.labels[idx]

    def load_image(self, image_path):
        image = decode_image(image_path, mode=ImageReadMode.RGB)
        return image
