import glob
import sys
import os
from torch.utils.data import DataLoader
from Code.Training import dataset, transforms
import torchvision.transforms as trn
sys.path.append(os.path.abspath(os.path.join('..', 'dataset.py')))
sys.path.append(os.path.abspath(os.path.join('..', 'transforms.py')))
sys.path.append(os.path.abspath(os.path.join('..', 'maker.py')))

# https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
# https://realpython.com/storing-images-in-python/
# https://github.com/fab-jul/hdf5_dataloader


def get_hdf5_dataset(data_path, model_name, folder, batch_size=1, max_chuncks=None):
    # create transform
    # Note: cannot use default PyTorch ops, because they expect PIL Images
    transform_hdf5 = trn.Compose([transforms.ArrayToTensor()])

    # create dataset
    all_file_ps = glob.glob(f'{data_path}/{folder}/*.hdf5')
    ds = dataset.HDF5Dataset(all_file_ps, transform=transform_hdf5, model_name=model_name, max_chuncks=max_chuncks)

    # using the standard PyTorch DataLoader
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4)

    print("Num Images in Dataset:", len(ds))
    print("Example Image and Label:", ds[0])
    for image_batch, label_batch in dl:
        print("Image and Label Batch Size:", image_batch.size(), label_batch.size())
        break

    return dl


def main():
    data_path = '../../Test_Data'
    get_hdf5_dataset(data_path, 'dias_model', 'Validation')


if __name__ == "__main__":
    main()
