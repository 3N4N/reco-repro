from torch.utils.data import Dataset, DataLoader



def get_pascal_dataloader(root_dir, split='train', batch_size=8, num_workers=4, input_size=513):
    dataset = PascalVOCSegmentation(
        root_dir=root_dir,
        split=split,
        input_size=input_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
        
    )
    
    return dataloader, dataset