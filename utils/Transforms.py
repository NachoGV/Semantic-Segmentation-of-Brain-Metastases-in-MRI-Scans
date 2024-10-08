import torch
import nibabel as nib
from scipy.ndimage import zoom
from monai.utils import set_determinism
from monai.transforms import MapTransform
from nilearn.datasets import load_mni152_template
from monai.transforms import (
    Compose,
    Spacingd,
    Zoomd,
    RandFlipd,
    LoadImaged,
    AsDiscrete,
    RandAffined,
    EnsureTyped,
    Activations,
    Orientationd,
    RandRotate90d,
    RandSpatialCropd,
    RandGaussianNoised,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureChannelFirstd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandScaleIntensityd,
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
    
    def train_ensemble(self, spatial_size):
            
        crop = RandSpatialCropd(keys=["image", "label"], roi_size=[240, 240, 160], random_size=False)
        crop.set_random_state(self.seed)
        self.flip_0.set_random_state(self.seed)
        self.flip_1.set_random_state(self.seed)

        return Compose(
            [
            Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest")
                    ),
            crop,
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
            self.flip_0,
            self.flip_1,
            ]
        )
    
    def val_ensemble(self):
        return Compose(
            [
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            ]
        )

    def tta_ensemble(self):
        return [
            ##############
            Compose([
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
            ]),
            ##############
            Compose([
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
                RandFlipd(keys=["image", "label"], prob=1, spatial_axis=0, allow_missing_keys=True),
            ]),
            ##############
            Compose([
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
                RandFlipd(keys=["image", "label"], prob=1, spatial_axis=1, allow_missing_keys=True),
            ]),
            ##############
            Compose([
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
                RandRotate90d(keys=["image", "label"], prob=1, max_k=1, spatial_axes=(0, 1), allow_missing_keys=True),
            ]),
            ##############
            Compose([
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
                Zoomd(keys=["image", "label"], zoom=1.1, mode=("bilinear", "nearest"), allow_missing_keys=True, keep_size=True),
            ]),
            ##############
            Compose([
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
                Zoomd(keys=["image", "label"], zoom=0.9, mode=("bilinear", "nearest"), allow_missing_keys=True, keep_size=True),
            ]),
            ##############
            Compose([
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
                RandAffined(keys=["image", "label"], prob=1, 
                            rotate_range=(0.1, 0.1, 0),
                            scale_range=(0.1, 0.1, 0),
                            mode=("bilinear", "nearest"), 
                            allow_missing_keys=True),
            ]),
            ##############
            Compose([
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
                RandGaussianNoised(keys=["image"], prob=1, mean=0, std=0.1, allow_missing_keys=True),
            ]),
            ##############
            Compose([
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
                RandAdjustContrastd(keys=["image"], prob=1, gamma=(0.9, 1.1), allow_missing_keys=True),
            ]),
            ##############
            Compose([
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
                RandGaussianSmoothd(keys=["image"], prob=1, sigma_x=(0.5, 1.0), allow_missing_keys=True),
            ]),
            ##############
            Compose([
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
                RandScaleIntensityd(keys=["image"], prob=1, factors=0.1, allow_missing_keys=True),
            ]),
        ]
    
# To MNI Space 
def mni_transform(img):
    # Template to match
    mni_template = load_mni152_template()

    # Resize
    zoom_factors = [t_dim / s_dim for t_dim, s_dim in zip(mni_template.shape, img.shape)]
    img_resized = zoom(img.get_fdata(), zoom_factors, order=0)
    img_resized = nib.Nifti1Image(img_resized, mni_template.affine)

    return img_resized