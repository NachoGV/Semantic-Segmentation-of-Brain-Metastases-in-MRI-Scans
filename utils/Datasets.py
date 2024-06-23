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
        
# Model Majority Voting Emsemble Dataset
class Ensemble_Dataset(CacheDataset):
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
                m_img.append(torch.load(channel))
            images.append(torch.stack(m_img, dim = 0))

        # Ensemble - Weighted Average
        images = torch.stack(images, dim=0)

        print(images.shape)

        majority_votes, _ = torch.mode(images, dim=0)

        # Label
        label = []
        for channel in self.labels[index]:
            label.append(torch.tensor(nib.load(channel).get_fdata()).unsqueeze(0))
        label = torch.cat(label, dim = 0)

        # Transforms
        images = self.transform({'image': images})['image'].float()
        label = self.transform({'image': label})['image'].float()
        
        return images, label
        
    def __len__(self):
        return len(self.images)