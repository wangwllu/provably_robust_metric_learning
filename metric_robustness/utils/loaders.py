from abc import ABC, abstractmethod

import os
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import numpy as np
from typing import List

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN


torch.multiprocessing.set_sharing_strategy('file_system')


def encode_labels(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test


def scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


class Loader(ABC):

    def __init__(self, root, random, seed):
        self._root = root
        self._random = random
        self._seed = seed

    def __call__(self):
        X_train, y_train, X_test, y_test = self._load_preprocessed_data()
        if self._random:
            X_test, y_test = shuffle(X_test, y_test, random_state=self._seed)
        return X_train, y_train, X_test, y_test

    @abstractmethod
    def _load_preprocessed_data(self):
        pass


class LibsvmLoader(Loader):

    def _load_preprocessed_data(self):
        X_train, y_train, X_test, y_test = load_svmlight_files(
            [self._path_of_train, self._path_of_test]
        )

        X_train = X_train.toarray()
        X_test = X_test.toarray()

        y_train, y_test = encode_labels(y_train, y_test)
        X_train, X_test = scale_features(X_train, X_test)

        return X_train, y_train, X_test, y_test

    @property
    def _path_of_train(self):
        return os.path.join(self._root, self._file_name_of_train)

    @property
    def _path_of_test(self):
        return os.path.join(self._root, self._file_name_of_test)

    @property
    @abstractmethod
    def _file_name_of_train(self):
        pass

    @property
    @abstractmethod
    def _file_name_of_test(self):
        pass


class GisetteLoader(LibsvmLoader):

    @property
    def _file_name_of_train(self):
        return 'gisette_scale'

    @property
    def _file_name_of_test(self):
        return 'gisette_scale.t'


class LetterLoader(LibsvmLoader):

    @property
    def _file_name_of_train(self):
        return 'letter.scale'

    @property
    def _file_name_of_test(self):
        return 'letter.scale.t'


class PendigitsLoader(LibsvmLoader):

    @property
    def _file_name_of_train(self):
        return 'pendigits'

    @property
    def _file_name_of_test(self):
        return 'pendigits.t'


class UspsLoader(LibsvmLoader):

    @property
    def _file_name_of_train(self):
        return 'usps'

    @property
    def _file_name_of_test(self):
        return 'usps.t'


class SatimageLoader(LibsvmLoader):

    @property
    def _file_name_of_train(self):
        return 'satimage.scale'

    @property
    def _file_name_of_test(self):
        return 'satimage.scale.t'


class DnaLoader(LibsvmLoader):

    @property
    def _file_name_of_train(self):
        return 'dna.scale'

    @property
    def _file_name_of_test(self):
        return 'dna.scale.t'


class SpliceLoader(LibsvmLoader):

    @property
    def _file_name_of_train(self):
        return 'splice'

    @property
    def _file_name_of_test(self):
        return 'splice.t'


class MnistLoader(Loader):

    @property
    def _torch_dataset_func(self):
        return MNIST

    def _load_preprocessed_data(self):
        X_train, y_train = self._load_train_dataset()
        X_test, y_test = self._load_test_dataset()
        return X_train, y_train, X_test, y_test

    def _load_train_dataset(self):
        torch_dataset = self._torch_dataset_func(
            root=self._root,
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )
        return self._load_from_torch_dataset(torch_dataset)

    def _load_test_dataset(self):
        torch_dataset = self._torch_dataset_func(
            root=self._root,
            train=False,
            transform=transforms.ToTensor(),
            download=True
        )
        return self._load_from_torch_dataset(torch_dataset)

    def _load_from_torch_dataset(self, torch_dataset):
        torch_loader = DataLoader(torch_dataset, batch_size=64, num_workers=4)
        X_list = []
        y_list = []
        for X, y in torch_loader:
            X_list.append(X)
            y_list.append(y)
        X_result = torch.cat(X_list, dim=0)
        y_result = torch.cat(y_list, dim=0)
        return X_result.view(X_result.shape[0], -1).numpy(), y_result.numpy()


class FashionMnistLoader(MnistLoader):

    @property
    def _torch_dataset_func(self):
        return FashionMNIST


class Cifar10Loader(MnistLoader):

    @property
    def _torch_dataset_func(self):
        return CIFAR10


# train -> split
class SvhnLoader(MnistLoader):

    @property
    def _torch_dataset_func(self):
        return SVHN

    def _load_train_dataset(self):
        torch_dataset = self._torch_dataset_func(
            root=self._root,
            split='train',
            transform=transforms.ToTensor(),
            download=True
        )
        return self._load_from_torch_dataset(torch_dataset)

    def _load_test_dataset(self):
        torch_dataset = self._torch_dataset_func(
            root=self._root,
            split='test',
            transform=transforms.ToTensor(),
            download=True
        )
        return self._load_from_torch_dataset(torch_dataset)


class LoaderDecorator(Loader):

    def __init__(self, decorated_loader: Loader):
        self._decorated_loader = decorated_loader

    def __call__(self):
        return self._decorated_loader()

    def _load_preprocessed_data(self):
        return self._decorated_loader._load_preprocessed_data()


class PartialLoaderDecorator(LoaderDecorator):

    def __init__(self, decorated_loader: Loader, label_domain: List[int]):
        super().__init__(decorated_loader)
        self._label_domain = label_domain

    def __call__(self):
        X_train, y_train, X_test, y_test = super()()

        X_train, y_train = self._select(X_train, y_train)
        X_test, y_test = self._select(X_test, y_test)

        y_train, y_test = encode_labels(y_train, y_test)
        return X_train, y_train, X_test, y_test

    def _select(self, X, y):
        mask = np.isin(y, self._label_domain)
        return X[mask], y[mask]


class LoaderFactory:

    def create(
            self, name, root, random=True, seed=None,
            partial=False, label_domain=None
    ):

        if name == 'gisette':
            loader = GisetteLoader(root, random, seed)
        elif name == 'letter':
            loader = LetterLoader(root, random, seed)
        elif name == 'pendigits':
            loader = PendigitsLoader(root, random, seed)
        elif name == 'usps':
            loader = UspsLoader(root, random, seed)
        elif name == 'satimage':
            loader = SatimageLoader(root, random, seed)
        elif name == 'dna':
            loader = DnaLoader(root, random, seed)
        elif name == 'splice':
            loader = SpliceLoader(root, random, seed)
        elif name == 'mnist':
            loader = MnistLoader(root, random, seed)
        elif name == 'fashion':
            loader = FashionMnistLoader(root, random, seed)
        elif name == 'cifar10':
            loader = Cifar10Loader(root, random, seed)
        elif name == 'svhn':
            loader = SvhnLoader(root, random, seed)
        else:
            raise Exception('unsupported dataset')

        if partial:
            assert label_domain is not None
            loader = PartialLoaderDecorator(loader, label_domain)

        return loader
