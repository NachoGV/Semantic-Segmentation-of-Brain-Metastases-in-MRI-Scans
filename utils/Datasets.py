import torch
import numpy as np
from monai.data import MetaTensor
from monai.data import CacheDataset

# Custos UCSF Dataset
class UCSF_Dataset(CacheDataset):
    def __init__(self, images, labels, transform, size = None):
        self.images = list(zip(*images))
        self.labels = list(labels)
        self.transform = transform

        if size is not None:
            self.images = self.images[:size]
            self.labels = self.labels[:size]

    def __getitem__(self, index):
        sample = {'image':[self.images[index][0], self.images[index][1], self.images[index][2], self.images[index][3]], 'label':self.labels[index]}
        return self.transform(sample)

    def __len__(self):
        return len(self.images)
    
# Custom Ensemble Dataset
class EnsembleDataset(CacheDataset):
    def __init__(self, images, transform = None, size = None, include_unet = True, model_name = 'logreg', tta = False):
        self.images = images
        self.transform = transform 
        self.include_unet = include_unet
        self.model_name = model_name
        self.tta = tta

        if size is not None:
            self.images = self.images[:size]

    def __getitem__(self, index):
        # Image & Label
        load_ahnet = self.images.iloc[index]['AHNet']
        load_segresnet = self.images.iloc[index]['SegResNet']
        load_unet = self.images.iloc[index]['UNet']
        load_unetr = self.images.iloc[index]['UNETR']
        load_label = self.images.iloc[index]['GT']
            
        # Load Images and Labels
        ahnet_image = [np.load(x)['arr_0'] for x in load_ahnet]
        segresnet_image = [np.load(x)['arr_0'] for x in load_segresnet]
        unet_image = [np.load(x)['arr_0'] for x in load_unet]
        unetr_image = [np.load(x)['arr_0'] for x in load_unetr]
        img_label = [np.load(x)['arr_0'] for x in load_label] 

        # To Tensor
        identity_affine = np.eye(4)  
        ahnet_image = [MetaTensor(torch.from_numpy(x), affine=identity_affine) for x in ahnet_image]
        segresnet_image = [MetaTensor(torch.from_numpy(x), affine=identity_affine) for x in segresnet_image]
        unet_image = [MetaTensor(torch.from_numpy(x), affine=identity_affine) for x in unet_image]
        unetr_image = [MetaTensor(torch.from_numpy(x), affine=identity_affine) for x in unetr_image]
        img_label = [MetaTensor(torch.from_numpy(x), affine=identity_affine) for x in img_label]

        # Stack Images and Label
        ahnet_image = torch.stack(ahnet_image, dim = 0)
        segresnet_image = torch.stack(segresnet_image, dim = 0)
        unet_image = torch.stack(unet_image, dim = 0)
        unetr_image = torch.stack(unetr_image, dim = 0)
        img_label = torch.stack(img_label, dim = 0)

        og_shape = ahnet_image.shape

        # TTA
        if self.tta:
            assert self.transform is not None; 'Transforms must be defined for TTA'
            imgs = [ahnet_image, segresnet_image, unet_image, unetr_image]
            imgs = torch.cat(imgs, dim = 0)
            ahnet_image, segresnet_image, unet_image, unetr_image = [], [], [], []
            for transform in self.transform:
                trasnformed = transform({'image': imgs, 'label':img_label})
                t_imgs = trasnformed['image']
                ahnet_image.append(t_imgs[:3])
                segresnet_image.append(t_imgs[3:6])
                unet_image.append(t_imgs[6:9])
                unetr_image.append(t_imgs[9:12])
            images = []
            for i in range(len(ahnet_image)):
                imgs_i = None
                if self.include_unet:
                    imgs_i = [ahnet_image[i], segresnet_image[i], unet_image[i], unetr_image[i]]
                else:
                    imgs_i = [ahnet_image[i], segresnet_image[i], unetr_image[i]]
                # Flatten
                if self.model_name == 'logreg':
                    imgs_i = [x.contiguous().view(3, -1).t() for x in imgs_i]
                    imgs_i = torch.cat(imgs_i, dim=1)
                elif self.model_name == 'conv3d':
                    imgs_i = torch.cat(imgs_i, dim=0)
                else:
                    print('Invalid Model Name')
                    return None
                images.append(imgs_i)

            if self.model_name == 'logreg':
                img_label = img_label.contiguous().view(3, -1).t()
                
            return images, img_label, og_shape

        # NOT TTA
        else: 
            if self.transform is not None:
                imgs = [ahnet_image, segresnet_image, unet_image, unetr_image]
                imgs = torch.cat(imgs, dim = 0)
                trasnformed = self.transform({'image': imgs, 'label': img_label})
                imgs = trasnformed['image']
                img_label = trasnformed['label']
                ahnet_image = imgs[:3]
                segresnet_image = imgs[3:6]
                unet_image = imgs[6:9]
                unetr_image = imgs[9:12]

                # Images
                images = []
                if not self.include_unet:
                    images = [ahnet_image, segresnet_image, unetr_image]
                else:
                    images = [ahnet_image, segresnet_image, unet_image, unetr_image]
                    
                # Flatten
                if self.model_name == 'logreg':
                    images = [x.contiguous().view(3, -1).t() for x in images]
                    images = torch.cat(images, dim=1)
                    img_label = img_label.contiguous().view(3, -1).t()
                elif self.model_name == 'conv3d':
                    images = torch.cat(images, dim=0)
                else:
                    print('Invalid Model Name')
                    return None
                
                return images, img_label, og_shape
        
    def __len__(self):
        return len(self.images)