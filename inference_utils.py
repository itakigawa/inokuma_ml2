import timm
import numpy as np
import torch
import cv2
from skimage import io
from torch.utils.data import Dataset
from timm.models.layers import SelectAdaptivePool2d
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TestImageDataset(Dataset):
    def __init__(self, img_paths, transform=None, auto_brightness=True):
        self.transform = transform
        self.img_paths = img_paths
        self.auto_brightness = auto_brightness

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        x = io.imread(img_path)
        if self.auto_brightness:
            x = auto_brightness(x)
        if self.transform:
            augmented = self.transform(image=x)
            x = augmented["image"]

        return x

def auto_brightness(img, min_brightness=0.6, max_value=255):
    brightness = np.sum(img) / (max_value * img.shape[0] * img.shape[1])
    ratio = brightness / min_brightness
    bright_img = cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)
    return bright_img

def tta_predict(model, device, dataset, transform, num_tta, softmax):
    y_pred = []
    model.eval()
    with torch.no_grad():
        for img in dataset:
            images = torch.stack(
                [transform(image=img)["image"] for i in range(num_tta)]
            )
            images = images.to(device)
            out = model(images)
            pred_out = out.mean(dim=0).detach().cpu()
            if softmax == False:
                pred_out = pred_out.softmax(dim=-1)
            y_pred.append(pred_out.numpy())
    return np.array(y_pred)

def get_da_filters(cfg):
    cent_crop_size = cfg["center_crop_size"]
    crop_size = cfg["crop_size"]
    input_size = cfg["input_size"]
    da_filters = [
        A.CenterCrop(cent_crop_size, cent_crop_size),
        A.RandomCrop(crop_size, crop_size),
        A.Resize(input_size, input_size),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    if cfg["normalize"]:
        da_filters.append(A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        )) 
    else:
        da_filters.append(A.ToFloat(max_value=255.0))
    da_filters.append(ToTensorV2()),
    return A.Compose(da_filters)

class CustomModel(nn.Module):
    def __init__(self, timm_model, gpool, head):
        super(CustomModel, self).__init__()
        self.backbone = timm_model
        self.gpool = gpool
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.forward_features = self.backbone.forward_features
        self.head = head

    def forward(self, x):
        f = self.forward_features(x)
        if hasattr(self.backbone, "global_pool"):
            if self.gpool:
                f = self.gpool(f)
            else:
                f = self.backbone.global_pool(f)
        y = self.head(f)
        return y

def create_net(
    model_name, head="bestfitting", concat_pool=False, outdim=1, softmax=True
):
    model_timm = timm.create_model(model_name, pretrained=True)
    num_ftrs = model_timm.num_features
    if concat_pool:
        neck = SelectAdaptivePool2d(output_size=1, pool_type="catavgmax", flatten=True)
        num_ftrs *= 2
    else:
        neck = SelectAdaptivePool2d(output_size=1, pool_type="avg", flatten=True)
    if head == "bestfitting":
        layers = [
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, int(num_ftrs / 2)),
            nn.ReLU(),
            nn.BatchNorm1d(int(num_ftrs / 2)),
            nn.Dropout(p=0.5),
            nn.Linear(int(num_ftrs / 2), outdim),
        ]
    elif head == "bn_linear":
        layers = [
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, outdim),
        ]
    else:
        layers = [nn.Dropout(p=0.5), nn.Linear(num_ftrs, outdim)]
    if softmax:
        layers.append(nn.Softmax(dim=-1))
    clf = nn.Sequential(*layers)
    model = CustomModel(model_timm, neck, clf)

    return model

def get_model_from_file(cfg, fname):
    model = create_net(
      cfg["model_name"], cfg["head"], cfg["concat_pool"], cfg["outdim"], cfg["softmax"]
    )
    model.load_state_dict(torch.load(fname))
    return model