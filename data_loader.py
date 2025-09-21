from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

DATASET_NAME = "flwrlabs/fed-isic2019"
BATCH_SIZE   = 16

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

def get_augmented_transform():
    """
    Returns augmentation transforms suitable for medical images (ISIC dataset).
    Uses conservative augmentations to preserve medical image integrity.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),  # Small rotation for skin lesions
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip is safe for skin lesions
        transforms.ColorJitter(
            brightness=0.1,  # Slight brightness variation
            contrast=0.1,    # Slight contrast variation
            saturation=0.1,  # Slight saturation variation
            hue=0.05         # Very small hue variation
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),  # Small translation
            scale=(0.95, 1.05),      # Small scale changes
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

def make_loader(hf_ds, batch_size=BATCH_SIZE, shuffle=False, use_augmentation=True):
    """
    Create a DataLoader from a HuggingFace dataset.
    
    Args:
        hf_ds: HuggingFace dataset
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        use_augmentation: If True, doubles the dataset by adding augmented versions
    """
    if use_augmentation:
        # Create augmented dataset by concatenating original with augmented version
        orig_tfm = get_transform()
        aug_tfm = get_augmented_transform()
        
        # Process original dataset
        def prep_original(batch):
            imgs = [orig_tfm(img) for img in batch["image"]]
            return {"pixel_values": imgs, "labels": batch["label"]}
        
        # Process augmented dataset  
        def prep_augmented(batch):
            imgs = [aug_tfm(img) for img in batch["image"]]
            return {"pixel_values": imgs, "labels": batch["label"]}
        
        # Create two separate datasets
        ds_orig = hf_ds.map(prep_original, batched=True)
        ds_aug = hf_ds.map(prep_augmented, batched=True)
        
        # Set format for both
        ds_orig.set_format(type="torch", columns=["pixel_values", "labels"])
        ds_aug.set_format(type="torch", columns=["pixel_values", "labels"])
        
        # Concatenate the datasets
        from datasets import concatenate_datasets
        ds = concatenate_datasets([ds_orig, ds_aug])
        print(f"Original: {len(ds_orig)}")
        print(f"Augmented: {len(ds)}")

    else:
        # Use only original transform (for test data)
        tfm = get_transform()
        def prep(batch):
            imgs = [tfm(img) for img in batch["image"]]
            return {"pixel_values": imgs, "labels": batch["label"]}
        ds = hf_ds.map(prep, batched=True)
        ds.set_format(type="torch", columns=["pixel_values", "labels"])
    
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def prepare_federated_loaders(num_clients):
    fds = FederatedDataset(
        dataset=DATASET_NAME,
        partitioners={
            "train": NaturalIdPartitioner(partition_by="center"),
            "test":  NaturalIdPartitioner(partition_by="center"),
        }
    )
    train_loaders = [
        make_loader(fds.load_partition(cid, "train"), shuffle=True, use_augmentation=True)
        for cid in range(num_clients)
    ]
    test_loaders  = [
        make_loader(fds.load_partition(cid, "test"), shuffle=False, use_augmentation=False)
        for cid in range(num_clients)
    ]

    global_test = load_dataset(DATASET_NAME, split="test")
    global_loader = make_loader(global_test, shuffle=False, use_augmentation=False)


    return train_loaders, test_loaders, global_loader

def count_centers():
    ds = load_dataset(DATASET_NAME, split="train")
    return len(ds.unique("center"))
