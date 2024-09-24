import os

import json
import requests
import zipfile
import torchvision
from configs.config_global import DATA_DIR

def objects_dataset_prepare():
    obj_data_dir = os.path.join(DATA_DIR, 'objects')
    if not os.path.isdir(obj_data_dir):
        os.makedirs(obj_data_dir)

        # Download the objects dataset
        url = 'https://bradylab.ucsd.edu/stimuli/ObjectsAll.zip'
        r = requests.get(url, allow_redirects=True)
        filename = os.path.join(obj_data_dir, url.rsplit('/', 1)[1])
        open(filename, 'wb').write(r.content)

        # Unzip the objects dataset
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(obj_data_dir)


def image_dataset_prepare():
    data = torchvision.datasets.MNIST(root=DATA_DIR, download=True)
    data = torchvision.datasets.CIFAR10(root=DATA_DIR, download=True)
    data = torchvision.datasets.CIFAR100(root=DATA_DIR, download=True)

if __name__ == "__main__":
    # auditory_data_prepare()
    # objects_dataset_prepare()
    image_dataset_prepare()
