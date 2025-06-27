from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import numpy as np

def split_dataset(dataset, train_ratio=0.5, attack_ratio=0.3):
    indices = np.random.permutation(len(dataset))
    train_end = int(len(dataset) * train_ratio)
    attack_end = train_end + int(len(dataset) * attack_ratio)

    return{
        'train': dataset[indices[:train_end]],
        'attack': dataset[indices[train_end:attack_end]],
        'test': dataset[indices[attack_end:]]
    }

if __name__=="__main__":
    dataset = TUDataset(root='data/', name='ENZYMES')
    splits = split_dataset(dataset)

    for key in splits:
        print(f"{key} set size: {len(splits[key])}")
