import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from configs.config_global import DATA_DIR
from PIL import Image
import numpy as np


class ObjectsDataset(Dataset):
    """
    Objects dataset from Brady et al. (2008)
    contain 2400 images of different objects
    """
    def __init__(self):
        obj_data_dir = os.path.join(DATA_DIR, 'objects', 'OBJECTSALL')
        if not os.path.isdir(obj_data_dir):
            raise NotImplementedError("Objects data set not available, "
                                      "run prepare_dataset.py")
        self.main_dir = obj_data_dir
        file_list = os.listdir(self.main_dir)
        # filter out non-image files
        self.all_imgs = [f for f in file_list if (f[-4:] == '.jpg' or f[-4:] == '.JPG')]

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc)
        label = idx
        return image, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = ObjectsDataset()
    print(len(dataset))
    image = np.array(dataset[20][0])
    print(image.shape)
    plt.imshow(image)
    plt.show()


