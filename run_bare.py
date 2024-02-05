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
        for lib in [np, pd, matplotlib, torch, timm, A, skimage, hydra, mlflow]:
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

    warmup_out = f"{outdir}/best_warmup.model"
    uptrain_out = f"{outdir}/best_uptrain.model"
    #cheating_out = f"{outdir}/cheating.model"

    writer = SummaryWriter(f"{outdir}/tblog")

    criterion, softmax = setup_loss(cfg["loss"])

    for k in cfg.keys():
        logging.info(f"{k} = {cfg[k]}")
        mlflow.log_param(k, cfg[k])

    result_summary = {}
    target_info = {}

    data, outdim = define_data(cfg, target_info)

    cent_crop_size = cfg["center_crop_size"]
    crop_size = cfg["crop_size"]
    input_size = cfg["input_size"]
   
    tr_filters = []
    if cfg["preproc"]:
        tr_filters.extend([
		        A.CenterCrop(cent_crop_size, cent_crop_size),
            #A.RandomCrop(crop_size, crop_size),
            A.Resize(input_size, input_size),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])
    else:
        tr_filters = [
 		        A.CenterCrop(cent_crop_size, cent_crop_size),
            #A.RandomCrop(crop_size, crop_size),
            A.Resize(input_size, input_size),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
         ]

    if cfg["normalize"]:
        tr_filters.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            )
        )
    else:
        tr_filters.append(A.ToFloat(max_value=255.0))
    tr_filters.append(ToTensorV2())
    train_da = A.Compose(tr_filters)

    print('train_da', train_da, file=sys.stderr)

    vl_filters = [
 		        A.CenterCrop(cent_crop_size, cent_crop_size),
            #A.RandomCrop(crop_size, crop_size),
            A.Resize(input_size, input_size),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
    ]
    if cfg["normalize"]:
        vl_filters.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            )
        )
    else:
        vl_filters.append(A.ToFloat(max_value=255.0))
    vl_filters.append(ToTensorV2())
    eval_da = A.Compose(vl_filters)

    print('eval_da', eval_da, file=sys.stderr)

    #da_transform = get_da_filters(cfg)
    #train_da = da_transform
    #eval_da = da_transform
    train_loader, valid_loader, test_loader = get_dataloaders(data, cfg, train_transform=train_da, eval_transform=eval_da)

    if cfg["grayscale"]:
        in_chans = 1
    else:
        in_chans = 3

    if cfg["pretrain"] == "ImageNet":
        model = create_net(
            cfg["model_name"], cfg["head"], cfg["concat_pool"], outdim, softmax, in_chans, pretrained=True
        )
    else: # None or speified
        model = create_net(
            cfg["model_name"], cfg["head"], cfg["concat_pool"], outdim, softmax, in_chans, pretrained=False
        )
        if cfg["pretrain"] != "None":
            pretrain_model = to_absolute_path(cfg["pretrain"])
            model.load_state_dict(torch.load(pretrain_model))

    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"# params: {num_params:,}", file=sys.stderr)
    logging.info(f"# params: {num_params:,}")
    mlflow.log_param("num_params", num_params)

    train_loss_log = []
    valid_loss_log = []

    # TRAIN WARMUP
    if cfg["warmup_epochs"] > 0:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

        optimizer, scheduler = learn_setup("warmup", model, len(train_loader), cfg)
        warmup_loss = float("inf")
        for epoch in range(1, cfg["warmup_epochs"] + 1):
            print(f"--- EPOCH {epoch} ---", file=sys.stderr)
            train_loss = train_step(
                model,
                device,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                epoch,
                cfg["grad_accum"],
            )
            valid_loss = test(model, device, valid_loader, criterion)
            if valid_loss < warmup_loss:
                warmup_loss = valid_loss
                torch.save(model.state_dict(), warmup_out)
                mlflow.log_artifact(warmup_out)
            writer.add_scalar("train_loss_warmup", train_loss, epoch)
            writer.add_scalar("valid_loss_warmup", valid_loss, epoch)
            train_loss_log.append([epoch, train_loss])
            valid_loss_log.append([epoch, valid_loss])
            print(f"LOSS train:{train_loss:.5} valid:{valid_loss:.5}", file=sys.stderr)
            logging.info(
                f"[warmup {epoch}] LOSS train:{train_loss:.5} valid:{valid_loss:.5}"
            )

        print(f"=== warmup best_loss {warmup_loss} ===", file=sys.stderr)

    # TRAIN UPTRAIN
    for param in model.parameters():
        param.requires_grad = True

    best_loss = {"epoch": -1, "score": float("inf")}
    unknown_loss = {"epoch": -1, "score": float("inf")}
    optimizer, scheduler = learn_setup("uptrain", model, len(train_loader), cfg)
    for epoch in range(1, cfg["uptrain_epochs"] + 1):
        print(f"--- EPOCH {epoch} ---", file=sys.stderr)
        train_loss = train_step(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            epoch,
            cfg["grad_accum"],
        )
        valid_loss = test(model, device, valid_loader, criterion)
        writer.add_scalar("train_loss_uptrain", train_loss, epoch)
        writer.add_scalar("valid_loss_uptrain", valid_loss, epoch)
        train_loss_log.append([cfg["warmup_epochs"] + epoch, train_loss])
        valid_loss_log.append([cfg["warmup_epochs"] + epoch, valid_loss])
        print(f"LOSS train:{train_loss:.5} test:{valid_loss:.5}", file=sys.stderr)
        logging.info(
            f"[uptrain {epoch}] LOSS train:{train_loss:.5} valid:{valid_loss:.5}"
        )
        # save the best snapshot
        if valid_loss < best_loss["score"]:
            print(
                f"valid) loss improved {best_loss['score']:.5} --> {valid_loss:.5}",
                file=sys.stderr,
            )
            logging.info(
                f"[uptrain {epoch}] loss improved {best_loss['score']:.5} --> {valid_loss:.5}"
            )
            best_loss["score"] = valid_loss
            best_loss["epoch"] = epoch
            torch.save(model.state_dict(), uptrain_out)
            mlflow.log_artifact(uptrain_out)

    print(
        f"=== best snapshot at epoch {best_loss['epoch']}: {best_loss['score']} ===",
        file=sys.stderr,
    )
    logging.info(f"best snapshot at epoch {best_loss['epoch']}: {best_loss['score']}")

    with open(f"{outdir}/train_loss_log.out", "w") as f:
        for logval in train_loss_log:
            print(logval[0], logval[1], file=f)

    with open(f"{outdir}/valid_loss_log.out", "w") as f:
        for logval in valid_loss_log:
            print(logval[0], logval[1], file=f)

    # PREDICTION
    model = create_net(
        cfg["model_name"], cfg["head"], cfg["concat_pool"], outdim, softmax, in_chans
    )

    # this is needed for non-tta predict
    train_loader, valid_loader, test_loader = get_dataloaders(data, cfg, train_transform=eval_da, eval_transform=eval_da)

    # PREDICTION
    model.load_state_dict(torch.load(uptrain_out))
    model.to(device)

    train_predcsv = f"{outdir}/train_prediction.csv"
    valid_predcsv = f"{outdir}/valid_prediction.csv"
    test_predcsv = f"{outdir}/test_prediction.csv"

    print("calc train_pred", file=sys.stderr)
    train_pkl = f"{outdir}/train.pkl"
    train_pngs = [f"{outdir}/train_each.png", f"{outdir}/train_all.png"]

    y_true, y_pred = calc_pred(model, device, train_loader, criterion, softmax)
    pred_train = Prediction(data.train_imgs, y_true, y_pred, outdim, data.names)
    csv_export(pred_train, fname=train_predcsv)
    plot_prediction(pred_train, train_pngs, "blue")
    log_results(y_true, y_pred, "train", result_summary)
    with open(train_pkl, "wb") as f:
        pickle.dump(pred_train, f)
    mlflow.log_artifact(train_pkl)
    for fname in train_pngs:
        mlflow.log_artifact(fname)

    print("calc valid_pred", file=sys.stderr)
    valid_pkl = f"{outdir}/valid.pkl"
    valid_pngs = [f"{outdir}/valid_each.png", f"{outdir}/valid_all.png"]

    y_true, y_pred = calc_pred(model, device, valid_loader, criterion, softmax)
    pred_valid = Prediction(data.valid_imgs, y_true, y_pred, outdim, data.names)
    csv_export(pred_valid, fname=valid_predcsv)
    plot_prediction(pred_valid, valid_pngs, "green")
    log_results(y_true, y_pred, "valid", result_summary)
    with open(valid_pkl, "wb") as f:
        pickle.dump(pred_valid, f)
    mlflow.log_artifact(valid_pkl)
    for fname in valid_pngs:
        mlflow.log_artifact(fname)

    print("calc test_pred", file=sys.stderr)
    test_pkl = f"{outdir}/test.pkl"
    test_pngs = [f"{outdir}/test_each.png", f"{outdir}/test_all.png"]

    y_true, y_pred = calc_pred(model, device, test_loader, criterion, softmax)
    pred_test = Prediction(data.test_imgs, y_true, y_pred, outdim, data.names)
    csv_export(pred_test, fname=test_predcsv)
    plot_prediction(pred_test, test_pngs, "red")
    log_results(y_true, y_pred, "test", result_summary)
    with open(test_pkl, "wb") as f:
        pickle.dump(pred_test, f)
    mlflow.log_artifact(test_pkl)
    for fname in test_pngs:
        mlflow.log_artifact(fname)

    if cfg['tta']:
      train_predcsv = f"{outdir}/train_tta_prediction.csv"
      valid_predcsv = f"{outdir}/valid_tta_prediction.csv"
      test_predcsv = f"{outdir}/test_tta_prediction.csv"

      print("calc train_tta", file=sys.stderr)
      train_tta_pkl = f"{outdir}/train_tta.pkl"
      train_tta_pngs = [f"{outdir}/train_tta_each.png", f"{outdir}/train_tta_all.png"]

      # this is needed for tta_predict
      data_train, data_valid, data_test = get_datasets(data, cfg)

      y_true, y_pred = tta_predict(
          model, device, data_train, train_da, cfg["num_tta"], softmax
      )
      pred_train = Prediction(data.train_imgs, y_true, y_pred, outdim, data.names)
      csv_export(pred_train, fname=train_predcsv)
      plot_prediction(pred_train, train_tta_pngs, "blue")
      log_results(y_true, y_pred, "train_tta", result_summary)
      with open(train_tta_pkl, "wb") as f:
          pickle.dump(pred_train, f)
      mlflow.log_artifact(train_tta_pkl)
      for fname in train_tta_pngs:
          mlflow.log_artifact(fname)

      print("calc valid_tta", file=sys.stderr)
      valid_tta_pkl = f"{outdir}/valid_tta.pkl"
      valid_tta_pngs = [f"{outdir}/valid_tta_each.png", f"{outdir}/valid_tta_all.png"]

      y_true, y_pred = tta_predict(
          model, device, data_valid, train_da, cfg["num_tta"], softmax
      )
      pred_valid = Prediction(data.valid_imgs, y_true, y_pred, outdim, data.names)
      csv_export(pred_valid, fname=valid_predcsv)
      plot_prediction(pred_valid, valid_tta_pngs, "green")
      log_results(y_true, y_pred, "valid_tta", result_summary)
      with open(valid_tta_pkl, "wb") as f:
          pickle.dump(pred_valid, f)
      mlflow.log_artifact(valid_tta_pkl)
      for fname in valid_tta_pngs:
          mlflow.log_artifact(fname)

      print("calc test_tta", file=sys.stderr)
      test_tta_pkl = f"{outdir}/test_tta.pkl"
      test_tta_pngs = [f"{outdir}/test_tta_each.png", f"{outdir}/test_tta_all.png"]

      y_true, y_pred = tta_predict(
          model, device, data_test, train_da, cfg["num_tta"], softmax
      )
      pred_test = Prediction(data.test_imgs, y_true, y_pred, outdim, data.names)
      csv_export(pred_test, fname=test_predcsv)
      plot_prediction(pred_test, test_tta_pngs, "red")
      log_results(y_true, y_pred, "test_tta", result_summary)
      with open(test_tta_pkl, "wb") as f:
          pickle.dump(pred_test, f)
      mlflow.log_artifact(test_tta_pkl)
      for fname in test_tta_pngs:
          mlflow.log_artifact(fname)


    # OUTPUT SUMMARY
    print("-" * 10, file=sys.stderr)
    logging.info(outdir)

    for k in cfg.keys():
        logging.info(f"{k} = {cfg[k]}")

    for metric in result_summary.keys():
        mlflow.log_metric(metric, result_summary[metric])
        logging.info(f"{metric}: {result_summary[metric]:.5}")

    result_csv = f"{outdir}/result_summary.csv"
    df = pd.Series(
        {
            "output dir": os.getcwd(),
            "train data": target_info["train_dir"],
            "test data": target_info["test_dir"],
            "ynames": data.names,
            "# train images": len(data.train_imgs),
            "# valid images": len(data.valid_imgs),
            "# test images": len(data.test_imgs),
            "model_name": cfg["model_name"],
            "input_size": cfg["input_size"],
            "crop_size": cfg["crop_size"],
            "warmup_epochs": cfg["warmup_epochs"],
            "uptrain_epochs": cfg["uptrain_epochs"],
            "pretrain": cfg["pretrain"],
            "grayscale": cfg["grayscale"],
            "Train RMSE": result_summary["rmse_train"],
            "Train MAE": result_summary["mae_train"],
            "Train R2": result_summary["r2_train"],
            "Valid RMSE": result_summary["rmse_valid"],
            "Valid MAE": result_summary["mae_valid"],
            "Valid R2": result_summary["r2_valid"],
            "Test RMSE": result_summary["rmse_test"],
            "Test MAE": result_summary["mae_test"],
            "Test R2": result_summary["r2_test"],
        },
        name="value",
    )
    df.to_csv(result_csv)

    logging.info("train_dir: " + target_info["train_dir"])
    logging.info("test_dir: " + target_info["test_dir"])
    logging.info(f"num train images: {len(data.train_imgs)}")
    logging.info(f"num valid images: {len(data.valid_imgs)}")
    logging.info(f"num test images: {len(data.test_imgs)}")
    logging.info(f"working dir: {os.getcwd()}")
    #logging.info(f"device: {torch.cuda.get_device_name(device)}")

    # IMAGE OUTPUT
    if cfg["image_out"] is False:
        return

    train_predfile = f"{outdir}/train_prediction.pdf"
    valid_predfile = f"{outdir}/valid_prediction.pdf"
    test_predfile = f"{outdir}/test_prediction.pdf"

    print("image_out train_pred", file=sys.stderr)
    with open(train_pkl, "rb") as f:
        pred = pickle.load(f)
    print_out_images(
        pred.imgs,
        pred.y_true,
        pred.y_pred,
        pred.outdim,
        fname=train_predfile,
        img_per_page=16,
    )
    mlflow.log_artifact(train_predfile)

    print("image_out valid_pred", file=sys.stderr)
    with open(valid_pkl, "rb") as f:
        pred = pickle.load(f)
    print_out_images(
        pred.imgs,
        pred.y_true,
        pred.y_pred,
        pred.outdim,
        fname=valid_predfile,
        img_per_page=16,
    )
    mlflow.log_artifact(valid_predfile)

    print("image_out test_pred", file=sys.stderr)
    with open(test_pkl, "rb") as f:
        pred = pickle.load(f)
    print_out_images(
        pred.imgs,
        pred.y_true,
        pred.y_pred,
        pred.outdim,
        fname=test_predfile,
        img_per_page=16,
    )
    mlflow.log_artifact(test_predfile)


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
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info(f"mps device")
    else:
        device = "cpu"
    print('device', device)
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
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg["lr_init_warmup"], weight_decay=1e-8
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["lr_max_warmup"],
            steps_per_epoch=num,
            epochs=cfg["warmup_epochs"],
        )
    elif mode == "uptrain":
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg["lr_init_uptrain"], weight_decay=1e-8
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


def get_dataloaders(data, cfg, train_transform=None, eval_transform=None):
    train_imgs = data.train_imgs
    train_labels = data.train_labels
    valid_imgs = data.valid_imgs
    valid_labels = data.valid_labels
    test_imgs = data.test_imgs
    test_labels = data.test_labels

    data_train = MyDataset(
        train_imgs, train_labels, transform=train_transform, grayscale=cfg["grayscale"]
    )
    data_valid = MyDataset(
        valid_imgs, valid_labels, transform=eval_transform, grayscale=cfg["grayscale"]
    )
    data_test = MyDataset(
        test_imgs, test_labels, transform=eval_transform, grayscale=cfg["grayscale"]
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
    return data, outdim


def load_from_dirs(dirs):
    assert dirs != []
    list_imgs = []
    list_labels = []
    for d in dirs:
        print("Loading", d)
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
