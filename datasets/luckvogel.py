import torch
from torch.utils.data import Dataset

from configs.configs import LuckVogelConfig
from tasks.tasktools import random

# gray
lg_gray_color = torch.tensor([214, 214, 214]) / 255
# red, blue, green, purple, black
lg_colors = torch.tensor([[209, 55, 49],
                          [36, 88, 159],
                          [97, 169, 76],
                          [177, 49, 126],
                          [0, 0, 0]]) / 255

def check_overlap(pos1, pos2, p_size):
    x1, y1 = pos1
    x2, y2 = pos2
    if x2 - p_size[0] <= x1 <= x2 + p_size[0] \
            and y2 - p_size[1] <= y1 <= y2 + p_size[1]:
        return True
    else:
        return False

def generate_pos(image_size, patch_size, n_patch):
    """
        image_size: (channels, height, width) 3-dimensional image size
        patch_size: (height, width) two-dimensional patch size
        n_patch: number of patches in the array
    """
    pos_array = []
    attempts = 0
    while len(pos_array) < n_patch:
        attempts += 1

        if attempts > 100:
            # print("attempts over 100, empty array can start over")
            pos_array = []
            attempts = 0
        
        x = random.randint(image_size[1] - patch_size[0])
        y = random.randint(image_size[2] - patch_size[1])
        new_pos = (x, y)

        # check if there are overlaps
        flag = True
        for prev_pos in pos_array:
            if check_overlap(new_pos, prev_pos, patch_size):
                flag = False
                break
        
        if flag:
            pos_array.append(new_pos)
        
    return pos_array

class LuckVogelDataset(Dataset):
    """Luck Vogel dataset"""

    def __init__(self, config: LuckVogelConfig):
        """
        Args:
            num_patch (int): number of patches in a single image
        """

        self.img_size = config.img_size
        self.num_patch = config.num_patches
        if isinstance(self.num_patch, int):
            self.num_patch = (self.num_patch, )

        # TODO: there might be problems generating images using
        #  the current method when number of patches is larger
        #  than 12 (can be solved with larger image size)
        assert max(self.num_patch) <= 12

        self.lg_patch_size = config.lg_patch_size
        # self.lg_small_patch_size = config.lg_small_patch_size

        self.data_set_len = 5000
        self.mode = config.task_mode
        assert self.mode == 0, "In current version, only support vanilla mode for a faster dataloader"

        self.changable_attributes = config.changable_attributes

    def __len__(self):
        return self.data_set_len

    def generate_attributes(self, num_patch):
        change_idx = random.randint(num_patch)
        attr = []
        changed_attr = []

        for i in range(num_patch):
            color = random.randint(len(lg_colors))
            attr.append(color)
            changed_attr.append(color)
            
            if i == change_idx: # Randomly select a patch to a different color
                changed_color = random.randint(len(lg_colors) - 1)
                if changed_color >= color:
                    changed_color += 1
                changed_attr[-1] = changed_color

        return attr, changed_attr

    def paint(self, positions, attributes):
        image = lg_gray_color[0] * torch.ones(self.img_size)
        for pos, att in zip(positions, attributes):
            patch = lg_colors[att][:, None, None].\
                repeat(1, self.lg_patch_size[0], self.lg_patch_size[1])
            image[:, pos[0]: pos[0] + self.lg_patch_size[0], \
                        pos[1]: pos[1] + self.lg_patch_size[1]] = patch # paint the color the large patch

        return image


    """
    The following is the slower but more general version

    def generate_attributes(self, num_patch):
        # Generate attributes (color, arientation, etc.) for each patch.     
        configurations = (
            (len(lg_colors), ),
            (len(lg_colors), ) * 2,
        )
        config = configurations[self.mode]

        attr = [[] for _ in range(num_patch)]
        changed_attr = [[] for _ in range(num_patch)]
        change_idx = np.random.randint(num_patch)

        for i in range(num_patch):
            change_dim = random.choice(self.changable_attributes)

            for dim, lim in enumerate(config):
                attr[i].append(np.random.randint(lim))
                changed_attr[i].append(attr[i][-1])
                
                if change_idx == i and change_dim == dim:
                    while changed_attr[i][-1] == attr[i][-1]:
                        changed_attr[i][-1] = np.random.randint(lim)

        return attr, changed_attr

    def paint(self, positions, attributes):
        image = lg_gray_color[0] * torch.ones(self.img_size)
        for pos, att in zip(positions, attributes):
            if self.mode == 0 or self.mode == 1:  # color only
                patch = lg_colors[att[0]][:, None, None].\
                    repeat(1, self.lg_patch_size[0], self.lg_patch_size[1])
                image[:, pos[0]: pos[0] + self.lg_patch_size[0], \
                         pos[1]: pos[1] + self.lg_patch_size[1]] = patch # paint the color the large patch

                if self.mode == 1: # paint the color of the small patch
                    small_patch = lg_colors[att[1]][:, None, None].\
                        repeat(1, self.lg_small_patch_size[0], self.lg_small_patch_size[1])
                    x = pos[0] + (self.lg_patch_size[0] - self.lg_small_patch_size[0]) // 2
                    y = pos[1] + (self.lg_patch_size[1] - self.lg_small_patch_size[1]) // 2
                    image[:, x : x + self.lg_small_patch_size[0], y: y + self.lg_small_patch_size[1]] = small_patch
            else:
                raise ValueError(self.mode)

        return image
    """

    def __getitem__(self, idx):

        num_patch = self.num_patch[idx % len(self.num_patch)]
        positions = generate_pos(self.img_size, self.lg_patch_size, num_patch)
        attributes, changed_attributes = self.generate_attributes(num_patch)
        image = self.paint(positions, attributes)
        changed_image = self.paint(positions, changed_attributes)

        image, changed_image = torch.Tensor(image), torch.Tensor(changed_image)

        #if num_patch == 12:
        #    import matplotlib.pyplot as plt
        #    plt.imsave('image.png', image.permute(1, 2, 0).numpy())
        #    plt.imsave('image2.png', changed_image.permute(1, 2, 0).numpy())
        #    exit(0)

        return image, changed_image, num_patch

class LuckVogelClassificationDataset(LuckVogelDataset):

    def __init__(self, config: LuckVogelConfig, transform=None):
        super().__init__(config)
        self.transform = transform

    def __getitem__(self, idx):

        num_patch = self.num_patch[idx % len(self.num_patch)]
        positions = generate_pos(self.img_size, self.lg_patch_size, num_patch)
        attributes, _ = self.generate_attributes(num_patch)
        image = self.paint(positions, attributes)

        if self.transform is not None:
            image = self.transform(image)
            
        return image, num_patch - 1


if __name__ == '__main__':
    for i in range(10):
        print(generate_pos((3, 32, 32), (5, 5), 12))