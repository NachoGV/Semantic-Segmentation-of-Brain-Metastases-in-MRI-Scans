# Imports
import torch
import random
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from Transforms import Transforms
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from skimage.measure import regionprops, label
from monai.transforms import Activations, AsDiscrete

# Parameters
seed = 33
VAL_AMP = True
transforms = Transforms(seed)
channels = ['TC', 'WT', 'ET']  

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
    
# Generate and Save Predictions
def gen_predictions(model, model_name, dataloader, dataframe, spatial_size, mode):
    
        # Params
        subject_cont = 0
        device = config()
        folder = None
        if mode == 'train':
            folder = 'train'
        elif mode == 'val':
            folder = 'val'
        elif mode == 'test':
            folder = 'test'
        else:
            print('Invalid Mode')
            return

        # Model
        model.eval()

        # Infer
        with torch.no_grad():
            for data in tqdm(dataloader):

                # Load Data
                inputs, labels = (
                    data["image"].to(device),
                    data["label"].to(device),
                )

                # Infer Data
                outputs = inference(inputs, spatial_size, model)
                trans = Activations(sigmoid=True)
                outputs = trans(outputs)

                # Save
                for i, channel in enumerate(channels):
                    torch.save(labels[0][i], f'../outputs/gt_segs/{folder}_gt_segs/gt_{dataframe["SubjectID"][subject_cont]}_{channel}.pt')
                    torch.save(outputs[0][i], f'../outputs/{model_name}/pred_segs/{folder}_pred_segs/pred_{dataframe["SubjectID"][subject_cont]}_{channel}.pt')                    

                # Update Counter
                subject_cont += 1
   
# Ensemble Inference
def ensemble_inference(dataframe, ensemble_function, threshold = 0.5, apply_crf = False):

    # Transforms
    trans = AsDiscrete(threshold=threshold)

    # Dice Params
    dice_values, dice_values_tc, dice_values_wt, dice_values_et = [], [], [], []
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    # Biometrics Params
    ids = []
    gt_nm, pred_nm = {'TC': [], 'WT': [], 'ET': []}, {'TC': [], 'WT': [], 'ET': []}
    gt_v, pred_v = {'TC': [], 'WT': [], 'ET': []}, {'TC': [], 'WT': [], 'ET': []}

    # Iterate over the dataframe
    for i in range(len(dataframe)):

        # Subject & Label
        subject_id = dataframe['SubjectID'][i]
        load_label = dataframe['GT'][i]
        
        # Images
        load_ahnet = dataframe['AHNet'][i]
        load_segresnet = dataframe['SegResNet'][i]
        load_unet = dataframe['UNet'][i]
        load_unetr = dataframe['UNETR'][i]
        
        # Params 
        image_voxel_volume = np.prod((1,1,1))
        label_voxel_volume = np.prod((1,1,1))

        # Load Images and Labels
        ahnet_image = [torch.load(x) for x in load_ahnet]
        segresnet_image = [torch.load(x) for x in load_segresnet]
        unet_image = [torch.load(x) for x in load_unet]
        unetr_image = [torch.load(x) for x in load_unetr]
        img_label = [torch.load(x) for x in load_label] 
            
        # Stack Images and Label
        ahnet_image = torch.stack(ahnet_image, dim = 0).unsqueeze(0)
        segresnet_image = torch.stack(segresnet_image, dim = 0).unsqueeze(0)
        unet_image = torch.stack(unet_image, dim = 0).unsqueeze(0)
        unetr_image = torch.stack(unetr_image, dim = 0).unsqueeze(0)
        img_label = torch.stack(img_label, dim = 0).unsqueeze(0)

        # Ensemble Function
        img = None
        if apply_crf:
            img = ensemble_function(img_label, [ahnet_image, segresnet_image, unet_image, unetr_image])
        else:
            img = ensemble_function([ahnet_image, segresnet_image, unet_image, unetr_image])

        # Discretizise
        img = trans(img)
        img_label = trans(img_label)

        # Dice Metric
        dice_metric(y_pred=img, y=img_label)
        dice_score = dice_metric.aggregate()
        dice_values.append(dice_score.item())

        dice_metric.reset()
            
		# Batch Dice
        dice_metric_batch(y_pred=img, y=img_label)
        dice_batch = dice_metric_batch.aggregate()
        dice_values_tc.append(dice_batch[0].item())
        dice_values_wt.append(dice_batch[1].item())
        dice_values_et.append(dice_batch[2].item())
        dice_metric_batch.reset()     

        # Biometrics
        for j, channel in enumerate(channels):
            # Image
            props = regionprops(label(nib.Nifti1Image(img[0][j].cpu().numpy(), np.eye(4)).get_fdata()))
            volumes = [prop.area * image_voxel_volume for prop in props]
            pred_nm[channel].append(int(len(volumes)))
            pred_v[channel].append(int(np.sum(volumes)))
            # Label
            props = regionprops(label(nib.Nifti1Image(img_label[0][j].cpu().numpy(), np.eye(4)).get_fdata()))
            volumes = [prop.area * label_voxel_volume for prop in props]
            gt_nm[channel].append(int(len(volumes)))
            gt_v[channel].append(int(np.sum(volumes)))

        # Subject ID
        ids.append(subject_id)
                
    # Excel
    df = pd.DataFrame({
        'SubjectID': ids,
		'Dice': dice_values,
		'Dice TC': dice_values_tc,
		'Dice WT': dice_values_wt,
		'Dice ET': dice_values_et,
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
	})
    
    return df

# Individual model test set inferer
def calculate_metrics(model_name, dataframe, threshold = 0.5):

    trans = AsDiscrete(threshold=threshold)

    # Dice Params
    dice_values, dice_values_tc, dice_values_wt, dice_values_et = [], [], [], []
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    # Biometrics Params
    ids = []
    gt_nm, pred_nm = {'TC': [], 'WT': [], 'ET': []}, {'TC': [], 'WT': [], 'ET': []}
    gt_v, pred_v = {'TC': [], 'WT': [], 'ET': []}, {'TC': [], 'WT': [], 'ET': []}
    gt_paths, pred_paths = [], []

    # Iterate over the dataframe
    for i in range(len(dataframe)):
        
        # Load Data
        subject_id = dataframe['SubjectID'][i]
        load_image = dataframe[model_name][i]
        load_label = dataframe['GT'][i]
        img = [torch.load(x) for x in load_image]
        img_label = [torch.load(x) for x in load_label]
        img = torch.stack(img, dim = 0).unsqueeze(0)
        img_label = torch.stack(img_label, dim = 0).unsqueeze(0)

        # Discretizise
        img = trans(img)
        img_label = trans(img_label)

        # Dice Metric
        dice_metric(y_pred=img, y=img_label)
        dice_score = dice_metric.aggregate()
        dice_values.append(dice_score.item())
        dice_metric.reset()
            
		# Batch Dice
        dice_metric_batch(y_pred=img, y=img_label)
        dice_batch = dice_metric_batch.aggregate()
        dice_values_tc.append(dice_batch[0].item())
        dice_values_wt.append(dice_batch[1].item())
        dice_values_et.append(dice_batch[2].item())
        dice_metric_batch.reset()     

        # Biometrics
        image_voxel_volume = np.prod((1,1,1))
        label_voxel_volume = np.prod((1,1,1))
        for j, channel in enumerate(channels):
            # Image
            props = regionprops(label(nib.Nifti1Image(img[0][j].cpu().numpy(), np.eye(4)).get_fdata()))
            volumes = [prop.area * image_voxel_volume for prop in props]
            pred_nm[channel].append(int(len(volumes)))
            pred_v[channel].append(int(np.sum(volumes)))
            # Label
            props = regionprops(label(nib.Nifti1Image(img_label[0][j].cpu().numpy(), np.eye(4)).get_fdata()))
            volumes = [prop.area * label_voxel_volume for prop in props]
            gt_nm[channel].append(int(len(volumes)))
            gt_v[channel].append(int(np.sum(volumes)))

        # Paths
        load_image = [x[3:] for x in load_image]
        load_label = [x[3:] for x in load_label]
        gt_paths.append(load_label)
        pred_paths.append(load_image)

        # Subject ID
        ids.append(subject_id)
                
    # Excel
    df = pd.DataFrame({
        'SubjectID': ids,
		'Dice': dice_values,
		'Dice TC': dice_values_tc,
		'Dice WT': dice_values_wt,
		'Dice ET': dice_values_et,
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
        'Pred Paths': pred_paths,
        'GT Paths': gt_paths
	})
    
    return df