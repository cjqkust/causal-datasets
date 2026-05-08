import pandas as pd
import numpy as np
import torch
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import StandardScaler


class MyDataset(Dataset):

    def __init__(self, path, mask=0):
        # super().__init__(path, params)
        self.data = np.loadtxt(path, delimiter=',', dtype=np.float32)
        self.x_dim = self.data.shape[-1] - 3
        self.x_dim_start = int(mask*self.x_dim)
        self.x_dim -= self.x_dim_start
        # self.scaler = StandardScaler()
        # self.data = self.scaler.fit_transform(self.data)

    def __getitem__(self, index):

        return self.data[index, self.x_dim_start:]

    def __len__(self):

        return len(self.data)

    def get_sampler(self, treat_weight=1):

        t = self.data[:, -3].astype(np.int16)
        count = Counter(t)
        class_count = np.array([count[0], count[1]*treat_weight])
        weight = 1. / class_count
        samples_weight = torch.tensor([weight[item] for item in t])
        sampler = WeightedRandomSampler(
            samples_weight,
            len(samples_weight),
            replacement=True)

        return sampler



def acic2016_processor(exp_number='47', dup_number='323803659', random_state=1):

    # data_x = pd.read_csv('Datasets/ACIC/x.csv')
    data_x = pd.read_csv('Datasets/ACIC/x_numerical.csv')
    data_ty = pd.read_csv(f'Datasets/ACIC/{exp_number}/zymu_{dup_number}.csv').loc[:, ['z', 'mu0', 'mu1', "y0", "y1"]]
    data = data_x.merge(data_ty, right_index=True, left_index=True)
    data['yf'] = data['z'] * data['y1'] + (1 - data['z']) * data['y0']
    data['ycf'] = data['z'] * data['y0'] + (1 - data['z']) * data['y1']
    data['muf'] = data['z'] * data['mu1'] + (1 - data['z']) * data['mu0']
    data['mucf'] = data['z'] * data['mu0'] + (1 - data['z']) * data['mu1']

    data = data.drop(['mu0', 'mu1', "y0", "y1"], axis=1)
    data = data.to_numpy()
    train, eval_test = train_test_split(data, test_size=0.37, stratify=data[:, -5], random_state=random_state)
    evaluation, test = train_test_split(eval_test, test_size=0.27, stratify=eval_test[:, -5], random_state=random_state)
    np.savetxt("Datasets/ACIC/train.csv", train[:, :-2], delimiter=",")
    np.savetxt("Datasets/ACIC/traineval.csv", np.concatenate((train[:, :-4], train[:, [-2, -1]]), axis=-1), delimiter=",")
    np.savetxt("Datasets/ACIC/eval.csv", np.concatenate((evaluation[:, :-4], evaluation[:, [-2, -1]]), axis=-1), delimiter=",")
    np.savetxt("Datasets/ACIC/test.csv", np.concatenate((test[:, :-4], test[:, [-2, -1]]), axis=-1), delimiter=",")

    print('New ACIC2016 Data')
    return None


def ihdp_processor(exp_number=1):

    file_path = f'Datasets/IHDP/ihdp_npci_{exp_number}.csv'
    data = np.loadtxt(file_path, delimiter=',')
    t = data[:, [0]]
    yf = data[:, [1]]
    ycf = data[:, [2]]
    mu0 = data[:, [3]]
    mu1 = data[:, [4]]
    muf = mu1 * t + mu0 * (1-t)
    mucf = mu1 * (1-t) + mu0 * t
    x = data[:, 5:] 
    output = np.concatenate([x, t, yf, ycf, muf, mucf], axis=-1)
    # we use yf for training, and mu for test. It enables us to split the data in this way since the ground truth mu would not be used in training.
    train, eval_test = train_test_split(output, test_size=0.37, stratify=output[:, -5], random_state=42)
    evaluation, test = train_test_split(eval_test, test_size=0.27, stratify=eval_test[:, -5], random_state=42)
    np.savetxt(f"Datasets/IHDP/train_{exp_number}.csv", train[:, :-2], delimiter=",")
    np.savetxt(f"Datasets/IHDP/traineval_{exp_number}.csv", np.concatenate((train[:, :-4], train[:, [-2, -1]]), axis=-1), delimiter=",")
    np.savetxt(f"Datasets/IHDP/eval_{exp_number}.csv", np.concatenate((evaluation[:, :-4], evaluation[:, [-2, -1]]), axis=-1), delimiter=",")
    np.savetxt(f"Datasets/IHDP/test_{exp_number}.csv", np.concatenate((test[:, :-4], test[:, [-2, -1]]), axis=-1), delimiter=",")

    np.savetxt(f"Datasets/IHDP/train.csv", train[:, :-2], delimiter=",")
    np.savetxt(f"Datasets/IHDP/traineval.csv", np.concatenate((train[:, :-4], train[:, [-2, -1]]), axis=-1), delimiter=",")
    np.savetxt(f"Datasets/IHDP/eval.csv", np.concatenate((evaluation[:, :-4], evaluation[:, [-2, -1]]), axis=-1), delimiter=",")
    np.savetxt(f"Datasets/IHDP/test.csv", np.concatenate((test[:, :-4], test[:, [-2, -1]]), axis=-1), delimiter=",")

    return None


if __name__ == "__main__":

    acic2016_processor()
    [ihdp_processor(i) for i in range(1,11)]
