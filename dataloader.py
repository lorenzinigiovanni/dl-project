import torch
import os
from torchvision import transforms as T
from dataset import Market1501Dataset


def get_data(batch_size, test_batch_size):

    # Transformations applied to both training and testing
    transform = list()
    transform.append(T.ConvertImageDtype(torch.float))
    transform.append(T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    )
    transform.append(T.Resize((256, 128)))

    # Transformations applied to training only
    training_transform = list()
    training_transform.append(T.RandomRotation(5))
    training_transform.append(T.RandomCrop((256, 128), 10))
    training_transform.append(T.RandomHorizontalFlip())
    # training_transform.append(T.ColorJitter(
    #     brightness=0.5,
    #     contrast=0.5,
    #     saturation=0.5,
    #     hue=0.2
    # ))

    # Get data from the dataset
    full_training_data = Market1501Dataset(
        os.path.join("dataset", "train"),
        os.path.join("dataset", "annotations_train.csv"),
        transform=T.Compose(transform + training_transform),
    )

    test_data = Market1501Dataset(
        os.path.join("dataset", "test"),
        transform=T.Compose(transform),
    )
    
    query_data = Market1501Dataset(
        os.path.join("dataset", "queries"),
        transform=T.Compose(transform),
    )

    # Create train and validation splits
    num_samples = len(full_training_data)
    training_samples = int(num_samples*0.7+1)

    # Find the last person in the training set so to not cut data
    person_id = full_training_data.dict[training_samples][1]
    while (person_id == full_training_data.dict[training_samples][1]):
        training_samples += 1
    person_id = full_training_data.dict[training_samples][1]

    # Create training and validation subset
    training_data = torch.utils.data.Subset(
        full_training_data,
        list(range(0, training_samples)),
    )

    validation_data = torch.utils.data.Subset(
        full_training_data,
        list(range(training_samples, num_samples)),
    )

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = torch.utils.data.DataLoader(
        validation_data,
        test_batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        test_batch_size,
        shuffle=False,
        num_workers=0,
    )

    query_loader = torch.utils.data.DataLoader(
        query_data,
        test_batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader, query_loader, person_id
