import os
import posixpath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import tqdm
from torchvision import datasets, models, transforms


def load_data(data_path):
    images_list = []
    # traverse root directory, and list directories as dirs and files as files
    print("loading images...")
    for root, dirs, images in tqdm.tqdm(os.walk(data_path)):
        path = root.split(os.sep)
        print(root)
        print((len(path) - 1) * '---', os.path.basename(root))
        for image_name in images:
            print(len(path) * '---', image_name)
            image = plt.imread(posixpath.join(root, image_name))
            images_list.append(image)
    print("loading images done")

    images_list = np.array(images_list)
    print('Number of images in the dataset: {}'.format(len(images_list)))
    print('Each images size is: {}'.format(images_list.shape[1:]))
    print('These are the first 4 images:')

    fig, ax_array = plt.subplots(4, 4)
    for i, ax in enumerate(ax_array.flat):
        ax.imshow(images_list[i], cmap='gray')
        ax.set_yticks([])
        ax.set_xticks([])
    plt.show()

    return images_list

    # for data_path in paths:
    #     image = plt.imread(data_path)  # load
    #     image = (image - mean) / std  # normalized
    #
    #     data = torch.tensor(image.T).view(1, 3, 110, 110)
    #
    #     # print(data.size())
    #     data = data.to(device=device, dtype=torch.float)
    #     data = Variable(data.cuda(device))
    #
    #     output = model_dias(data)
    #     print(data_path + " - diastolic:")
    #     print(output.item())


def main():
    data_path = '../../Data'
    load_data(data_path)


if __name__ == "__main__":
    main()
