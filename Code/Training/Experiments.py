import argparse
import json
import os
import sys
import torch
import numpy as np
import tqdm.auto

from Code.Preprocessing.Project_B import Config as cfg

sys.path.append('../../Code/Training/Trainer.py')
sys.path.append('../../Code/Training')
sys.path.append('../../Code/Training/TrainResult.py')
sys.path.append('../../Code/Preprocessing/Project_B/Config.py')

# Now do your import
import ResNet, LoadData, HDF5DataLoader, DenseNet, PlotConfusion
from TrainResult import FitResult
from Trainer import Trainer
import datetime
from datetime import date

today = date.today().strftime("%d_%m_%Y")

def densenet_experiment(
    run_name,
    out_dir=cfg.DENSENET_RESULTS,
    #data_path='../../Data',
    data_path=cfg.DATASET_DIR,
    model_name='dias_model',
    seed=None,
    device=None,
    # Training params
    bs_train=128,
    bs_test=None,
    batches=100,
    epochs=100,
    early_stopping=3,
    checkpoints=cfg.DENSENET_CHECKPOINTS,
    lr=1e-3,
    weight_decay=1e-3,
    eps=1e-6,
    gamma=0.9,
    scheduler=False,
    plot_confusion=False
):
    if not seed:
        seed = torch.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoints:
        checkpoints = f"{checkpoints}/{today}_{model_name}.pt"
    cfg = locals()

    fit_res = None

    print("run on:", device)

    if plot_confusion is True:
        model_path = None
        model = torch.load(model_path)
        print(f"\n*** Load checkpoint {model_path}")
    else:
        model = DenseNet.create_densenet_model()
    model = model.to(device)

    loss_fn = torch.nn.L1Loss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if scheduler is True:
        # Decay learning rate each epoch
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=5, verbose=True)

    trainer = Trainer(model, loss_fn, optimizer, device, scheduler, model_name=model_name)

    # dl_train = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Train', batch_size=bs_train, max_chuncks=80)
    # dl_valid = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Validation', batch_size=bs_test, max_chuncks=20)
    dl_train = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Train', batch_size=bs_train, max_chuncks=16)
    dl_valid = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Validation', batch_size=bs_test, max_chuncks=4)
    assert len(dl_valid) == len(dl_train)

    if plot_confusion is True:
        epoch_res = trainer.test_epoch(dl_train, verbose=True, max_batches=batches, plot_confusion=True)
        PlotConfusion.confusion_matrix(epoch_res, save_dir=out_dir)

        save_experiment(run_name, out_dir, cfg, epoch_res, model_name)
    else:
        fit_res = trainer.fit(dl_train, dl_valid, num_epochs=epochs, print_every=1, early_stopping=early_stopping,
                              checkpoints=checkpoints, max_batches=batches, plot_confusion=False)

        save_experiment(run_name, out_dir, cfg, fit_res, model_name)


def resnet_experiment(
    run_name,
    #out_dir="../../Results/experiments/resnet18",
    # data_path='../../Data',
    out_dir=cfg.RESNET_RESULTS,
    data_path= cfg.DATASET_DIR,
    model_name='dias_model',
    seed=None,
    device=None,
    # Training params
    bs_train=128,
    bs_test=None,
    batches=100,
    epochs=100,
    early_stopping=3,
    checkpoints=cfg.RESNET_CHECKPOINTS,
    lr=1e-3,
    weight_decay=1e-3,
    eps=1e-6,
    gamma=0.9,
    scheduler=False,
    plot_confusion=False,
):
    if not seed:
        seed = torch.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoints:
        checkpoints = f"{checkpoints}/{today}_{model_name}"
    cfg = locals()

    fit_res = None

    print("run on:", device)

    if plot_confusion is True:
        # Insert model path to plot
        model_path = None
        assert model_path is not None
        model = torch.load(model_path)
        print(f"\n*** Load checkpoint {model_path}")
    else:
        print("create model resnet")
        model = ResNet.create_resnet_model()
    model = model.to(device)

    loss_fn = torch.nn.L1Loss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if scheduler is True:
        # Decay learning rate each epoch
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=5, verbose=True)

    trainer = Trainer(model, loss_fn, optimizer, device, scheduler, model_name)

    # dl_train = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Train', batch_size=bs_train, max_chuncks=80)
    # dl_valid = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Validation', batch_size=bs_test, max_chuncks=20)
    dl_train = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Train', batch_size=bs_train, max_chuncks=16)
    dl_valid = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Validation', batch_size=bs_test, max_chuncks=4)
    assert len(dl_valid) == len(dl_train)

    if plot_confusion is True:
        epoch_res = trainer.test_epoch(dl_train, verbose=True, max_batches=batches, plot_confusion=True)
        PlotConfusion.confusion_matrix(epoch_res, save_dir=out_dir)
    else:
        fit_res = trainer.fit(dl_train, dl_valid, num_epochs=epochs, print_every=1, early_stopping=early_stopping,
                              checkpoints=checkpoints, max_batches=batches, plot_confusion=False)

        save_experiment(run_name, out_dir, cfg, fit_res, model_name)


def save_experiment(run_name, out_dir, cfg, fit_res, model_name):
    del cfg['device']
    output = dict(config=cfg, results=fit_res)
    print(cfg)
    lr = cfg['lr']
    cfg_LK = (
        f'{today}_{model_name}_{lr}'
    )
    output_filename = f"{os.path.join(out_dir, run_name)}_{cfg_LK}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        print(output_filename)
        json.dump(output, f, indent=2)


def load_experiment(filename):
    with open(filename, "r") as f:
        output = json.load(f)

    config = output["config"]
    fit_res = FitResult(*output["results"])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description="Experiments")
    sp = p.add_subparsers(help="Sub-commands")

    # Experiment config
    sp_exp = sp.add_parser(
        "run-exp", help="Run experiment with a single " "configuration"
    )
    # sp_exp.set_defaults(subcmd_fn=densenet_experiment)
    sp_exp.set_defaults(subcmd_fn=resnet_experiment)
    sp_exp.add_argument(
        "--run-name", "-n", type=str, help="Name of run and output file", required=True
    )
    sp_exp.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        # default=cfg.DENSENET_RESULTS,
        default=cfg.RESNET_RESULTS,
        required=False,
    )
    sp_exp.add_argument(
        "--seed", "-s", type=int, help="Random seed", default=None, required=False
    )
    sp_exp.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device (default is autodetect)",
        default=None,
        required=False,
    )
    sp_exp.add_argument(
        "--model-name",
        "-m",
        type=str,
        help="Model name",
        default="dias_model",
        required=False,
    )
    # # Training
    sp_exp.add_argument(
        "--bs-train",
        type=int,
        help="Train batch size",
        default=128,
        metavar="BATCH_SIZE",
    )
    sp_exp.add_argument(
        "--bs-test", type=int, help="Test batch size", metavar="BATCH_SIZE"
    )
    sp_exp.add_argument(
        "--batches", type=int, help="Number of batches per epoch", default=100
    )
    sp_exp.add_argument(
        "--epochs", type=int, help="Maximal number of epochs", default=100
    )
    sp_exp.add_argument(
        "--early-stopping",
        type=int,
        help="Stop after this many epochs without " "improvement",
        default=3,
    )
    sp_exp.add_argument(
        "--checkpoints",
        type=str,
        help="Save model checkpoints to this file when test " "accuracy improves",
        # default=cfg.DENSENET_CHECKPOINTS,
        default=cfg.RESNET_CHECKPOINTS,
    )
    sp_exp.add_argument("--lr", type=float, help="Learning rate", default=1e-2)
    sp_exp.add_argument("--weight_decay", type=float, help="Weight decay", default=1e-3)
    sp_exp.add_argument("--eps", type=float, help="Epsilon", default=1e-6)
    sp_exp.add_argument("--gamma", type=float, help="Scheduler gamma", default=0.5)
    sp_exp.add_argument("--scheduler", type=bool, help="scheduler", default=False)
    sp_exp.add_argument("--plot-confusion", "-p", type=bool, help="Plot confusion matrix", default=False, required=False)

    parsed = p.parse_args()

    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == "__main__":
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f"*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}")
    subcmd_fn(**vars(parsed_args))