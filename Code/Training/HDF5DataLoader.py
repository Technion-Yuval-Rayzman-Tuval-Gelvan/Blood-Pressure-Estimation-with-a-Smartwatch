import glob
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'dataset.py')))
sys.path.append(os.path.abspath(os.path.join('..', 'transforms.py')))
sys.path.append(os.path.abspath(os.path.join('..', 'maker.py')))
import maker
import dataset
import transforms
from dataset import HDF5Dataset
from transforms import ArrayToTensor, ArrayCenterCrop
# from hdf5_dataloader.dataset import HDF5Dataset
# from hdf5_dataloader.transforms import ArrayToTensor, ArrayCenterCrop
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
# https://realpython.com/storing-images-in-python/
# https://github.com/fab-jul/hdf5_dataloader


def get_hdf5_dataset (data_path, model_name, folder):
    # create transform
    # Note: cannot use default PyTorch ops, because they expect PIL Images
    transform_hdf5 = transforms.Compose([ArrayToTensor()])

    # create dataset
    all_file_ps = glob.glob(f'{data_path}/{folder}_HDF5/*.hdf5')
    ds = HDF5Dataset(all_file_ps, transform=transform_hdf5, model_name=model_name)

    # using the standard PyTorch DataLoader
    dl = DataLoader(ds, batch_size=64, num_workers=8)

    print("Num Images in Dataset:", len(ds))
    print("Example Image and Label:", ds[2])
    for image_batch, label_batch in dl:
        print("Image and Label Batch Size:", image_batch.size(), label_batch.size())
        break

    return dl


def main():
    data_path = '../../Test_Data'
    get_hdf5_dataset(data_path, 'dias_model', 'Validation')


if __name__ == "__main__":
    main()
