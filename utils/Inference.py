# Imports
import os
import torch
import random
import pandas as pd
from Transforms import Transforms
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
import numpy as np
import nibabel as nib
from skimage.measure import regionprops, label

# Parameters
seed = 33
VAL_AMP = True
transforms = Transforms(seed)

# Metrics
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

# Configurations
def config():

    # Determinism
    random.seed(seed) # Random
    set_determinism(seed=seed) # Monai
    np.random.seed(seed) # Numpy
    torch.manual_seed(seed) # PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # CUDA

    # Check if CUDA is available
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device

# Inference Method
def inference(input, spatial_size, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=spatial_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

   
# Individual model test set inferer
def test_model(model, model_name, test_loader, test_df, spatial_size):
    
	# Params
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []
    pred_nm ={'TC': [], 'WT': [], 'ET': []}
    gt_nm ={'TC': [], 'WT': [], 'ET': []}
    pred_v ={'TC': [], 'WT': [], 'ET': []}
    gt_v = {'TC': [], 'WT': [], 'ET': []}
    gt_paths = {'TC': [], 'WT': [], 'ET': []}
    pred_paths = {'TC': [], 'WT': [], 'ET': []}
    channels = ['TC', 'WT', 'ET']    
    device = config()

	# Model
    model.eval()
    
	# Infer
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            
            # Load Data
            test_inputs, test_labels = (
				test_data["image"].to(device),
				test_data["label"].to(device),
			)
            
			# Infer Data
            test_outputs = inference(test_inputs, spatial_size, model)
            test_outputs = [transforms.post()(x) for x in test_outputs]
            
			# Mean Dice
            dice_metric(y_pred=test_outputs, y=test_labels)
            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            
			# Batch Dice
            dice_metric_batch(y_pred=test_outputs, y=test_labels)
            metric_batch = dice_metric_batch.aggregate()
            # TC
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            # WT
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            # ET
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            # Reset
            dice_metric.reset()
            dice_metric_batch.reset()
            #print(f'Dice: {metric:.4f} | Dice TC: {metric_tc:.4f} | Dice WT: {metric_wt:.4f} | Dice ET: {metric_et:.4f}')
            
			# Ground Truth Number of Metastases and Total Volume
            for i, channel in enumerate(channels):
                 gt_image = nib.Nifti1Image(test_labels[0][i].cpu().numpy(), np.eye(4))
                 voxel_volume = np.prod(gt_image.header.get_zooms())
                 props = regionprops(label(gt_image.get_fdata()))
                 volumes = [prop.area * voxel_volume for prop in props]
                 gt_nm[channel].append(int(len(volumes)))
                 gt_v[channel].append(int(np.sum(volumes)))
                 path_to_gt = f'outputs/gt_segmentations/gt_{test_df["SubjectID"][len(metric_values)-1]}_{channel}.nii.gz'
                 nib.save(gt_image, f'../{path_to_gt}')
                 gt_paths[channel].append(path_to_gt)

			# Predicted Number of Metastases and Total Volume
            for i, channel in enumerate(channels):
                 nifti_image = nib.Nifti1Image(test_outputs[0][i].cpu().numpy(), np.eye(4))
                 voxel_volume = np.prod(nifti_image.header.get_zooms())
                 props = regionprops(label(nifti_image.get_fdata()))
                 volumes = [prop.area * voxel_volume for prop in props]
                 pred_nm[channel].append(int(len(volumes)))
                 pred_v[channel].append(int(np.sum(volumes)))
                 path_to_pred = f'outputs/{model_name}/pred_segmentations/pred_{test_df["SubjectID"][len(metric_values)-1]}_{channel}.nii.gz'
                 nib.save(nifti_image, f'../{path_to_pred}')      
                 pred_paths[channel].append(path_to_pred)
                
    # Excel
    df = pd.DataFrame({
        'SubjectID': test_df['SubjectID'],
		'Dice': metric_values,
		'Dice TC': metric_values_tc,
		'Dice WT': metric_values_wt,
		'Dice ET': metric_values_et,
		'Pred NM TC': pred_nm['TC'],
		'Pred NM WT': pred_nm['WT'],
		'Pred NM ET': pred_nm['ET'],
		'GT NM TC': gt_nm['TC'],
		'GT NM WT': gt_nm['WT'],
		'GT NM ET': gt_nm['ET'],
		'Pred V TC': pred_v['TC'],
		'Pred V WT': pred_v['WT'],
		'Pred V ET': pred_v['ET'],
		'GT V TC': gt_v['TC'],
		'GT V WT': gt_v['WT'],
		'GT V ET': gt_v['ET'],
        'Pred TC': pred_paths['TC'],
        'Pred WT': pred_paths['WT'],
        'Pred ET': pred_paths['ET'],
        'GT TC': gt_paths['TC'],
        'GT WT': gt_paths['WT'],
        'GT ET': gt_paths['ET'],
	})
    df.set_index('SubjectID', inplace=True)
    
    return df