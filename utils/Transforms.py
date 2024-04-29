import torch
from monai.utils import set_determinism
from monai.transforms import MapTransform
from monai.transforms import (
    Invertd,
    Compose,
    Spacingd,
    RandFlipd,
    LoadImaged,
    AsDiscrete,
    EnsureTyped,
    Activations,
    AsDiscreted,
    Activationsd,
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
    def __init__(self, seed):
        # Train Transformations
        self.crop = RandSpatialCropd(keys=["image", "label"], roi_size=[240, 240, 160], random_size=False)
        self.flip_0 = RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0)
        self.flip_1 = RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1)
        #self.flip_2 = RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2)
        self.scale = RandScaleIntensityd(keys="image", factors=0.1, prob=1.0)
        self.shift = RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0)
        
        # Monai
        self.seed = seed
        set_determinism(seed=seed) # Monai
    
    def train(self, spatial_size):
        # Set determinism
        self.crop.set_random_state(self.seed)
        self.flip_0.set_random_state(self.seed)
        self.flip_1.set_random_state(self.seed)
        #self.flip_2.set_random_state(self.seed)
        self.scale.set_random_state(self.seed)
        self.shift.set_random_state(self.seed)
        
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
                           mode=("bilinear", "nearest")
                       ),
                       self.crop,
                       ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
                       self.flip_0,
                       self.flip_1,
                       #self.flip_2,
                       NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                       self.scale,
                       self.shift
                       ]
                )
                       
    def val(self):
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
        
    def post(self):
        return Compose(
            [
            Activations(sigmoid=True), 
            AsDiscrete(threshold=0.5),
            ]
        )