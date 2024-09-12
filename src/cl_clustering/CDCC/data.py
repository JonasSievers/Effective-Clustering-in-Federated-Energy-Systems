from torch.utils.data import Dataset
from torch.fft import fft

from CDCC.augmentations import DataTransform_T, DataTransform_F

class CustomTrainDataset(Dataset):
    def __init__(self, model_params, x_tensor):
        self.x_data = x_tensor
        self.x_data_f = fft(self.x_data).abs()

        # Data augmentation
        self.data_t, self.aug_t = DataTransform_T(self.x_data,model_params)
        self.data_f, self.aug_f = DataTransform_F(self.x_data_f,model_params)

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return self.data_t[idx], self.aug_t[idx], self.data_f[idx], self.aug_f[idx]
    

class CustomTestDataset(Dataset):
    def __init__(self, x_tensor):
        self.x_tensor = x_tensor

    def __len__(self):
        return self.x_tensor.shape[0]

    def __getitem__(self, idx):
        return self.x_tensor[idx]