import os
from PIL import Image
import numpy as np
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class CelebAMaskHQM(Dataset):
    def __init__(self, label_path, transform_label, mode):
        self.label_path = label_path
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.preprocess()

        if mode:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        n = len([name for name in os.listdir(self.label_path) if os.path.isfile(os.path.join(self.label_path, name))])
        for i in range(n):
            label_path = os.path.join(self.label_path, str(i) + '.png')
            if self.mode:
                self.train_dataset.append(label_path)
            else:
                self.test_dataset.append(label_path)
        print('Finished preprocessing the CelebA dataset (mask only)...')

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode else self.test_dataset
        label_path = dataset[index]
        label = Image.open(label_path)
        return self.transform_label(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


class CelebAMaskHQMLoader:
    def __init__(self, dataset_path, transform, batch_size, num_workers, mode):
        self.dataset_path = dataset_path
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = CelebAMaskHQM(dataset_path, transform, mode)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,
                                     sampler=InfiniteSamplerWrapper(self.dataset), num_workers=num_workers)

    def load(self):
        return self.dataloader
