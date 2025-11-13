import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        from pathlib import Path
        # Get data file from data directory
        data_path = Path(__file__).parent.parent.parent / "data" / "training_data.csv"
        self.data = np.genfromtxt(data_path, delimiter=',')

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        # Save scaler to models directory
        models_path = Path(__file__).parent.parent.parent / "models" / "scaler.pkl"
        pickle.dump(self.scaler, open(models_path, "wb")) #save to normalize at inference

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        # Use actual training data instead of dummy values
        n = self.normalized_data[idx, 0:-1]
        y = self.normalized_data[idx, [-1]]

        x_tensor = torch.from_numpy(n).float()
        y_tensor = torch.from_numpy(y).float()
        dict1 = {}
        dict1 = {'input': x_tensor, 'label': y_tensor}
        return dict1


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        train_size = int(0.8 * len(self.nav_dataset))
        test_size = len(self.nav_dataset) - train_size

        self.train_loader, self.test_loader = torch.utils.data.random_split(self.nav_dataset , [train_size, test_size])

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)

    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
