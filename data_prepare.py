from torch_geometric.datasets import TUDataset
datasets = {
    'msrc':TUDataset(root='data/',name='MSRC_9'),
    'enzymes':TUDataset(root='data/', name='ENZYMES')
}

print(f"MSRC_9 dataset size: {len(datasets['msrc'])}")
print(f"ENZYMES dataset size: {len(datasets['enzymes'])}")
