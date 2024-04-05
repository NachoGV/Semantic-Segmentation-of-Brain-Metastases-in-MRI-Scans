import torch
from monai.transforms import MapTransform
from monai.transforms import (
    Compose,
    Spacingd,
    RandFlipd,
    LoadImaged,
    EnsureTyped,
    Orientationd,
    RandSpatialCropd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd
)

# BraTS Annotations Transform
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
    
# Transforms
class Transforms():
    def train(spatial_size):
        return Compose(
            [
            LoadImaged(keys=["image", "label"], reader="NibabelReader"),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[240, 240, 160], random_size=False),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
    def val():
        return Compose(
            [
            LoadImaged(keys=["image", "label"], reader="NibabelReader"),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )