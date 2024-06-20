import torch
import nibabel as nib
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
    
# Model ensemble custom dataset
class Model_Ensemble_Dataset(CacheDataset):
    def __init__(self, images, labels, transform, size = None):
        self.images = list(zip(*images))
        self.labels = list(labels)
        self.transform = transform

        if size is not None:
            self.images = self.images[:size]
            self.labels = self.labels[:size]

    def __getitem__(self, index):

        # Images
        images = []
        for model_images in self.images[index]:
            m_img = []
            for channel in model_images:
                m_img.append(torch.tensor(nib.load(channel).get_fdata()).unsqueeze(0))
            images.append(torch.cat(m_img, dim = 0))	

        # Label
        label = []
        for channel in self.labels[index]:
            label.append(torch.tensor(nib.load(channel).get_fdata()).unsqueeze(0))
        label = torch.cat(label, dim = 0)

        # Remove unnecessary dimensions
        images = [img.squeeze(0) for img in images]
        label = label.squeeze(0)

        # Transforms
        tr_images = []
        for img in images:
            tr_images.append(self.transform({'image': img})['image'].float())
        label = self.transform({'image': label})['image'].float()
        
        return tr_images, label
        
# Model Majority Voting Emsemble Dataset
class Voting_Ensemble_Dataset(CacheDataset):
    def __init__(self, images, labels, transform, size = None):
        self.images = list(zip(*images))
        self.labels = list(labels)
        self.transform = transform

        if size is not None:
            self.images = self.images[:size]
            self.labels = self.labels[:size]

    def __getitem__(self, index):

        # Images
        images = []
        for model_images in self.images[index]:
            m_img = []
            for channel in model_images:
                m_img.append(torch.tensor(nib.load(channel).get_fdata()).unsqueeze(0))
            images.append(torch.cat(m_img, dim = 0))
        images = torch.stack(images, dim=0)
        majority_votes, _ = torch.mode(images, dim=0)

        # Label
        label = []
        for channel in self.labels[index]:
            label.append(torch.tensor(nib.load(channel).get_fdata()).unsqueeze(0))
        label = torch.cat(label, dim = 0)

        # Remove unnecessary dimensions
        images = majority_votes.squeeze(0)
        label = label.squeeze(0)

        # Transforms
        images = self.transform({'image': images})['image'].float()
        label = self.transform({'image': label})['image'].float()
        
        return images, label
        
    def __len__(self):
        return len(self.images)
       
# Channel Ensemble custom Dataset 
class Channel_Ensemble_Dataset(CacheDataset):
    def __init__(self, images, labels, transform, size = None):
        self.images =[[images[j][i] for j in range(len(images))] for i in range(len(images[0]))]
        self.labels = list(labels)
        self.transform = transform

        if size is not None:
            self.images = self.images[:size]
            self.labels = self.labels[:size]

    def __getitem__(self, index):

        # Images
        images = []
        for channel_images in self.images[index]:
            m_img = []
            for channel in channel_images:
                m_img.append(torch.tensor(nib.load(channel).get_fdata()).unsqueeze(0))
            images.append(torch.cat(m_img, dim = 0))

        # Label
        label = []
        for channel in self.labels[index]:
            label.append(torch.tensor(nib.load(channel).get_fdata()).unsqueeze(0))
        label = torch.cat(label, dim = 0)
        
        # Remove unnecessary dimensions
        images = [img.squeeze(0) for img in images]
        label = label.squeeze(0)

        # Transforms
        tr_images = []
        for img in images:
            tr_images.append(self.transform({'image': img})['image'].float())
        label = self.transform({'image': label})['image'].float()
        
        return tr_images, label

    def __len__(self):
        return len(self.images)