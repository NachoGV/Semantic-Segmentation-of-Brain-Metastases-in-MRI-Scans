from monai.data import CacheDataset

# Definition of custom dataset
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