import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from torchvision import datasets, models, transforms
from multiprocessing import Pool, Lock
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import shutil
import sys
import time
sys.path.append(os.path.abspath(os.path.join('..', 'Training.py')))
import Training

batch_size = 64
total_samples = 2500000


class CustomImageFolderDataset(Dataset):
    """Custom Image Loader dataset."""

    def __init__(self, root, folder, model_name, subset_len: int, offset=0, transform=None):
        """
        Args:
            root (string): Path to the images organized in a particular folder structure.
            transform: Any Pytorch transform to be applied
        """

        save_path = f"{root}/Images_paths/{folder}"

        """Once get images paths and save"""
        # Get all image paths from a directory
        if not os.path.exists(save_path):
            self.image_paths = glob(f"{root}/{folder}/*/*")
            Training.save_data(self.image_paths, save_path)

        self.image_paths = Training.load_data(save_path)
        self.image_paths = self.image_paths[offset:(offset+subset_len)]
        # Get the labels from the image paths
        if model_name == 'dias_model':
            self.labels = [x.split("_")[-1][:-4] for x in self.image_paths]
        else:
            assert(model_name == 'sys_model')
            self.labels = [x.split("_")[-2] for x in self.image_paths]
        # Create a dictionary mapping each label to a index from 0 to len(classes).
        self.label_to_idx = {x: i for i, x in enumerate(set(self.labels))}
        self.transform = transform
        self.subset_len = subset_len

    def __len__(self):
        # return length of dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # open and send one image and label
        assert(self.subset_len > idx >= 0)
        img_name = self.image_paths[idx]
        label = float(self.labels[idx])
        # start = time.time()
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        # print("open image time:", time.time() - start)
        return image, label


def get_dir_list(data_path):
    return os.listdir(data_path)


def print_some_images(data_loader):
    for images_batch, label_batch in data_loader:
        images = images_batch
        break
    print('Number of images in the dataset: {}'.format(len(images)))
    print('Each images size is: {}'.format(images.shape[1:]))
    print('These are the first 4 images:')
    images = images.detach().to("cpu").numpy()
    fig, ax_array = plt.subplots(4, 4)
    for i, ax in enumerate(ax_array.flat):
        ax.imshow(images[i][0], cmap='gray')
        ax.set_yticks([])
        ax.set_xticks([])
    plt.show()


def arrange_folders(data_path):
    # Make Train Validation and Test directories
    dir_names = ["Train", "Validation", "Test"]

    train_list, val_list, test_list = split_data(data_path)
    data_lists = [train_list, val_list, test_list]

    for i, list in enumerate(data_lists):
        name = dir_names[i]
        path = os.path.join(data_path, name)
        if not os.path.exists(path):
            os.mkdir(path)

        print(path)
        for patient in list:
            print(patient)
            shutil.move(os.path.join(data_path, patient), f"{path}/")


def split_data(data_path):
    dir_list = get_dir_list(data_path)
    dir_list = np.array(dir_list)

    n_patients = len(dir_list)

    # Generate a random generator with a fixed seed
    rand_gen = np.random.RandomState(0)

    # Generating a shuffled vector of indices
    indices = np.arange(n_patients)
    rand_gen.shuffle(indices)

    # Split the indices into 60% train / 20% validation / 20% test
    n_patients_train = int(n_patients * 0.6)
    n_patients_val = n_patients_train + int(n_patients * 0.2)
    train_indices = indices[:n_patients_train]
    val_indices = indices[n_patients_train:n_patients_val]
    test_indices = indices[n_patients_val:]

    train_list = dir_list[train_indices]
    validation_list = dir_list[val_indices]
    test_list = dir_list[test_indices]

    print(f"Total Patients: {n_patients}")
    print(f"Train Patients: {len(train_list)}")
    print(f"Validation Patients: {len(validation_list)}")
    print(f"Test Patients: {len(test_list)}")

    return train_list, validation_list, test_list


def get_dataset(data_path, model_name, folder, offset):
    t = transforms.Compose([transforms.Resize((110, 110)),
                            transforms.Grayscale(),
                            transforms.ToTensor()])
    dir = f"{data_path}/{folder}"
    print(f"{folder} Dataset. Load Path: {dir}")
    max_samples = total_samples
    new_offset = offset
    if folder == "Train":
        max_samples = int(total_samples * 0.6)
        new_offset = int(offset * 0.6)
    # elif folder == "Validation":
    #     max_samples = int(total_samples * 0.2)
    #     new_offset = int(offset * 0.2)
    else:
        max_samples = int(total_samples * 0.2)
        new_offset = int(offset * 0.2)

    print(f"New samples: {max_samples}, Offset: {new_offset}")
    dataset = CustomImageFolderDataset(root=data_path, folder=folder, transform=t, model_name=model_name,
                                       subset_len=max_samples, offset=new_offset)
    print("Num Images in Dataset:", len(dataset))
    print("Example Image and Label:", dataset[2])
    sampler = SubsetRandomSampler(list(range(len(dataset))))
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=sampler, pin_memory=True)
    for image_batch, label_batch in data_loader:
        print("Image and Label Batch Size:", image_batch.size(), label_batch.size())
        break

    return data_loader


def print_batch_size(data_loader):
    for image_batch, label_batch in data_loader:
        print(image_batch.size(), label_batch.size())
        break


def check_images(img_name):
    print(img_name)
    t = transforms.Compose([transforms.Resize((110, 110)),
                            transforms.Grayscale(),
                            transforms.ToTensor()])
    image = Image.open(img_name)
    image = t(image)


def main():
    """Paths"""
    # data_path = '/media/tuvalgelvan@staff.technion.ac.il/hd-21/Estimated-Blood-Presure-Project/Blood-Pressure' \
    #             '-Estimation-with-a-Smartwatch/Data'
    # data_path = '../../Test_Data'
    data_path = '../../Data'

    """Split images to Test/Val/Test folders
       Activate only if all patients in the same directory"""
    # arrange_folders(data_path)

    """Check how to get image from folder"""
    # test_image_paths = glob(f"{data_path}/Test/*/*")
    # dias_labels = [x.split("_")[-1][:-4] for x in test_image_paths]
    # sys_labels = [x.split("_")[-2] for x in test_image_paths]
    # print(f"Image Path Example: {test_image_paths[0]}")
    # print(f"Dias Label Example: {dias_labels[0]}")
    # print(f"Sys Label Example: {sys_labels[0]}")

    """Get data example"""
    # dias_train_loader = get_dataset(data_path, 'dias_model', "Train")

    # print_some_images(dias_train_loader)

    """Debug to find images with errors"""
    # image_paths = glob(f"{data_path}/Validation/*/*")
    # # start_idx = int((len(image_paths)/100)*56)
    # image_paths = image_paths[(2411741+632177):]
    # pool = Pool()
    # for _ in tqdm.tqdm(pool.imap(func=check_images, iterable=image_paths), total=len(image_paths)):
    #     pass

    # for image in image_paths:
    #     check_images(image)

    image_paths = glob(f"{data_path}/Test/*/*")
    start_idx = int((len(image_paths) / 100) * 56)
    image_paths = image_paths[(1063270+1211404):]
    pool = Pool()
    for _ in tqdm.tqdm(pool.imap(func=check_images, iterable=image_paths), total=len(image_paths)):
        pass

    # for image in image_paths:
    #     check_images(image)


if __name__ == "__main__":
    main()
