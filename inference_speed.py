import datetime
import logging
import math
import os
import pickle
import re
import sys
import time
from operator import itemgetter

import albumentations as A
import glob
import hydra
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import skimage
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from hydra.utils import to_absolute_path
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import io
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import (
    BalancedCE_MSELoss,
    Data,
    MyDataset,
    Prediction,
    SymmetricSoftTargetCrossEntropy,
    calc_pred,
    create_net,
    load_data,
    include_imgs,
    plot_prediction,
    csv_export,
    print_out_images,
    test,
    train_step,
    tta_predict,
)


@hydra.main(config_name="config.yaml")
def main(cfg):
    with mlflow.start_run() as mlrun:
        t0 = time.time()
        run(cfg)
        elapsed = time.time() - t0

        logging.info(f"MLflow run id: {mlrun.info.run_id}")
        # versions
        for lib in [np, pd, matplotlib, torch, timm, A, hydra, mlflow]:
          logging.info(f"version {lib.__name__}: {lib.__version__}")
        logging.info(f"Elapsed time: {datetime.timedelta(seconds=elapsed)}")
        print(f"MLflow run id: {mlrun.info.run_id}", file=sys.stderr)
        print(f"Elapsed time: {datetime.timedelta(seconds=elapsed)}", file=sys.stderr)


def run(cfg):
    outdir = prepare_outdir()
    device = prepare_device()

    logging.info(f"workdir = {os.getcwd()}")
    logging.info(f"outdir = {outdir}")
    mlflow.log_param("outdir", outdir)

    writer = SummaryWriter(f"{outdir}/tblog")
    criterion, softmax = setup_loss(cfg["loss"])

    result_summary = {}
    target_info = {}

    data, outdim = define_data(cfg, target_info)
    da_transform = get_da_filters(cfg)
    train_loader, valid_loader, test_loader = get_dataloaders(data, da_transform, cfg)

    if cfg["grayscale"]:
        in_chans = 1
    else:
        in_chans = 3

    if cfg["pretrain"] == "ImageNet" or cfg["pretrain"] == "None":
      print('specify pretrained model')
      sys.exit(-1)
    else: 
        model = create_net(
            cfg["model_name"], cfg["head"], cfg["concat_pool"], outdim, softmax, in_chans, pretrained=False
        )
        pretrain_model = to_absolute_path(cfg["pretrain"])
        model.load_state_dict(torch.load(pretrain_model))

    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"# params: {num_params:,}", file=sys.stderr)
    logging.info(f"# params: {num_params:,}")
    mlflow.log_param("num_params", num_params)

    # PREDICTION
    data_train, data_valid, data_test = get_datasets(data, cfg)

    # PREDICTION

    print('tta {}'.format(cfg["num_tta"]))
    logging.info('tta {}'.format(cfg["num_tta"]))
    t0 = time.time()
    y_true, y_pred = tta_predict(
        model, device, data_test, da_transform, cfg["num_tta"], softmax
    )
    elapsed = time.time() - t0
    num = len(y_true)
    print(f"(GPU) Elapsed time ({num} image): {datetime.timedelta(seconds=elapsed)}", file=sys.stderr)
    print(f"(GPU) Elapsed time (per image): {datetime.timedelta(seconds=elapsed/num)}", file=sys.stderr)
    logging.info(f"(GPU) Elapsed time ({num} image): {datetime.timedelta(seconds=elapsed)}")
    logging.info(f"(GPU) Elapsed time (per image): {datetime.timedelta(seconds=elapsed/num)}")

    print('tta 1')
    logging.info('tta 1')
    t1 = time.time()
    y_true, y_pred = tta_predict(
        model, device, data_test, da_transform, 1, softmax
    )
    elapsed = time.time() - t1
    num = len(y_true)
    print(f"(GPU) Elapsed time ({num} image): {datetime.timedelta(seconds=elapsed)}", file=sys.stderr)
    print(f"(GPU) Elapsed time (per image): {datetime.timedelta(seconds=elapsed/num)}", file=sys.stderr)
    logging.info(f"(GPU) Elapsed time ({num} image): {datetime.timedelta(seconds=elapsed)}")
    logging.info(f"(GPU) Elapsed time (per image): {datetime.timedelta(seconds=elapsed/num)}")

    print('tta 1')
    logging.info('tta 1')
    t2 = time.time()
    y_true, y_pred = tta_predict(
        model, 'cpu', data_test, da_transform, 1, softmax
    )
    elapsed = time.time() - t2
    num = len(y_true)
    print(f"(CPU) Elapsed time ({num} image): {datetime.timedelta(seconds=elapsed)}", file=sys.stderr)
    print(f"(CPU) Elapsed time (per image): {datetime.timedelta(seconds=elapsed/num)}", file=sys.stderr)
    logging.info(f"(CPU) Elapsed time ({num} image): {datetime.timedelta(seconds=elapsed)}")
    logging.info(f"(CPU) Elapsed time (per image): {datetime.timedelta(seconds=elapsed/num)}")


def prepare_outdir():
    # output dir
    outdir = "output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    return outdir


def prepare_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_id = torch.cuda.current_device()
        logging.info(f"cuda device: {torch.cuda.get_device_name(cuda_id)}")
    else:
        device = "cpu"
    return device


def setup_loss(loss):
    if loss in ["SoftTargetCrossEntropy", "SymmetricSoftTargetCrossEntropy"]:
        criterion = getattr(sys.modules[__name__], loss)()
        softmax = False
    elif loss == "BalancedCE_MSELoss":
        criterion = BalancedCE_MSELoss(alpha=0.5)
        softmax = False
    elif loss in ["MSELoss", "L1Loss"]:
        criterion = getattr(nn, loss)()
        softmax = True
    else:
        print("loss is undefined")
    return criterion, softmax


def learn_setup(mode, model, num, cfg):
    if mode == "warmup":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg["lr_init_warmup"], weight_decay=0
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["lr_max_warmup"],
            steps_per_epoch=num,
            epochs=cfg["warmup_epochs"],
        )
    elif mode == "uptrain":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg["lr_init_uptrain"], weight_decay=0
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["lr_max_uptrain"],
            steps_per_epoch=num,
            epochs=cfg["uptrain_epochs"],
        )
    else:
        print("error")

    return optimizer, scheduler


def get_dataloaders(data, da_transform, cfg):
    train_imgs = data.train_imgs
    train_labels = data.train_labels
    valid_imgs = data.valid_imgs
    valid_labels = data.valid_labels
    test_imgs = data.test_imgs
    test_labels = data.test_labels

    data_train = MyDataset(
        train_imgs, train_labels, transform=da_transform, grayscale=cfg["grayscale"]
    )
    data_valid = MyDataset(
        valid_imgs, valid_labels, transform=da_transform, grayscale=cfg["grayscale"]
    )
    data_test = MyDataset(
        test_imgs, test_labels, transform=da_transform, grayscale=cfg["grayscale"]
    )

    train_loader = DataLoader(data_train, batch_size=cfg["batch_size"], shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(data_test, batch_size=cfg["batch_size"], shuffle=False)

    return train_loader, valid_loader, test_loader


def get_datasets(data, cfg):
    train_imgs = data.train_imgs
    train_labels = data.train_labels
    valid_imgs = data.valid_imgs
    valid_labels = data.valid_labels
    test_imgs = data.test_imgs
    test_labels = data.test_labels

    data_train = MyDataset(
        train_imgs, train_labels, transform=None, grayscale=cfg["grayscale"]
    )
    data_valid = MyDataset(
        valid_imgs, valid_labels, transform=None, grayscale=cfg["grayscale"]
    )
    data_test = MyDataset(
        test_imgs, test_labels, transform=None, grayscale=cfg["grayscale"]
    )
    return data_train, data_valid, data_test


def get_da_filters(cfg):
    cent_crop_size = cfg["center_crop_size"]
    crop_size = cfg["crop_size"]
    input_size = cfg["input_size"]
    da_filters = [
        A.CenterCrop(cent_crop_size, cent_crop_size),
        A.RandomCrop(crop_size, crop_size),
        A.Resize(input_size, input_size),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
        A.RandomRotate90(p=0.5)
    ]
    if cfg["flip"]:
        da_filters.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])
    if cfg["normalize"]:
        da_filters.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            )
        )
    else:
        da_filters.append(A.ToFloat(max_value=255.0))
    da_filters.append(ToTensorV2()),
    print(da_filters, file=sys.stderr)
    logging.info(da_filters)
    return A.Compose(da_filters)


def define_data(cfg, info):
    for key in ["train_dir", "val_dir", "test_dir", "val_size"]:
        info[key] = cfg[key]

    train_dir = to_absolute_path(cfg["train_dir"])
    test_dir = to_absolute_path(cfg["test_dir"])
    val_size = cfg["val_size"]
    if include_imgs(train_dir):
        given_imgs, given_labels, tag = load_from_dirs([train_dir])
    else:
        train_dirs = glob.glob(f"{train_dir}/*/")
        given_imgs, given_labels, tag = load_from_dirs(train_dirs)
    test_imgs, test_labels, tag = load_from_dirs([test_dir])
    if cfg["no_validation"]:
        valid_imgs, valid_labels = given_imgs, given_labels
        train_imgs, train_labels = given_imgs, given_labels
    elif cfg["val_dir"] == "None":
        train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(
            given_imgs, given_labels, test_size=val_size
        )
    else:
        val_dir = to_absolute_path(cfg["val_dir"])
        valid_imgs, valid_labels, tag = load_from_dirs([val_dir])
        train_imgs, train_labels = given_imgs, given_labels
    outdim, ynames = tag
    data = Data(
        train_imgs,
        train_labels,
        valid_imgs,
        valid_labels,
        test_imgs,
        test_labels,
        ynames,
    )
    print("# train images:", len(data.train_imgs))
    print("# valid images:", len(data.valid_imgs))
    print("# test images:", len(data.test_imgs))
    logging.info("# train images: {}".format(len(data.train_imgs)))
    logging.info("# valid images: {}".format(len(data.valid_imgs)))
    logging.info("# test images: {}".format(len(data.test_imgs)))
    return data, outdim


def load_from_dirs(dirs):
    assert dirs != []
    list_imgs = []
    list_labels = []
    for d in dirs:
        print("Loading", d)
        logging.info(f"Loading {d}")
        imgs, labels, ynames = load_data(d)
        outdim = len(ynames)
        list_imgs.extend(imgs)
        list_labels.extend(labels)
    for t in list_labels:
        assert outdim == len(t)
    return list_imgs, list_labels, (outdim, ynames)


def retrieve(train_dirs, valid_dirs, test_dirs):
    train_imgs, train_labels, tag = load_from_dirs(train_dirs)
    valid_imgs, valid_labels, tag = load_from_dirs(valid_dirs)
    test_imgs, test_labels, tag = load_from_dirs(test_dirs)
    outdim, ynames = tag
    print(len(train_imgs), "images loaded from", train_dirs)
    print(len(valid_imgs), "images loaded from", valid_dirs)
    print(len(test_imgs), "images loaded from", test_dirs)
    print("ynames", ynames)
    print("outdim", outdim)
    return (
        Data(
            train_imgs,
            train_labels,
            valid_imgs,
            valid_labels,
            test_imgs,
            test_labels,
            ynames,
        ),
        outdim,
    )


def log_results(y_true, y_pred, tag, result_summary):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    result_summary[f"rmse_{tag}"] = rmse
    result_summary[f"mae_{tag}"] = mae
    result_summary[f"r2_{tag}"] = r2

    print(f"[{tag}] rmse {rmse:.5} mae {mae:.5} r2 {r2:.5}", file=sys.stderr)
    logging.info(f"[{tag}] rmse {rmse:.5} mae {mae:.5} r2 {r2:.5}")


if __name__ == "__main__":
    s = f"file://{os.getcwd()}/mlruns"
    mlflow.set_tracking_uri(s)
    main()
