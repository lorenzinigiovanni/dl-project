import torch
import torchvision
from torchvision import transforms as T
import torch.nn.functional as F
import torch.nn as nn
import os
import pandas as pd
from Market1501Dataset import Market1501Dataset

# Reti presidenziali
from CanaleCinque import CanaleCinque


def get_optimizer(net, lr, wd, momentum):
    # optimizer = torch.optim.SGD(
    #     net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    return optimizer


def train(net, data_loader, optimizer, device='cuda:0'):
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    net.train()
    for (inputs, targets) in data_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)

        samples += inputs.shape[0]

        loss = net.get_loss(outputs, targets)
        cumulative_loss += loss.item()

        predicteds = {}
        for key, value in outputs.items():
            _, predicteds[key] = value.max(1)

        accuracy = net.get_accuracy(predicteds, targets)
        cumulative_accuracy += accuracy

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return cumulative_loss/samples, cumulative_accuracy/samples*100


def test(net, data_loader, device='cuda:0'):
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    net.eval()
    with torch.no_grad():
        for (inputs, targets) in data_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)

            samples += inputs.shape[0]

            loss = net.get_loss(outputs, targets)
            cumulative_loss += loss.item()  # Il .item() estrae uno scalare da un tensore

            predicteds = {}
            for key, value in outputs.items():
                _, predicteds[key] = value.max(1)

            accuracy = net.get_accuracy(predicteds, targets)
            cumulative_accuracy += accuracy

    return cumulative_loss/samples, cumulative_accuracy/samples*100


def get_data(batch_size, test_batch_size=256):

    transform = list()
    transform.append(T.ConvertImageDtype(torch.float))
    transform = T.Compose(transform)

    # Load data
    full_training_data = Market1501Dataset(
        os.path.join("dataset", "train"),
        os.path.join("dataset", "annotations_train.csv"),
        transform=transform
    )
    test_data = Market1501Dataset(
        os.path.join("dataset", "test"),
        transform=transform
    )

    # Create train and validation splits
    num_samples = len(full_training_data)
    training_samples = int(num_samples*0.7+1)

    person_id = full_training_data.dict[training_samples][1]
    # Sta l'ultima persona
    while (person_id == full_training_data.dict[training_samples][1]):
        training_samples += 1

    person_id = full_training_data.dict[training_samples][1]

    training_data = torch.utils.data.Subset(
        full_training_data, list(range(0, training_samples)))
    validation_data = torch.utils.data.Subset(
        full_training_data, list(range(training_samples, num_samples)))

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        validation_data, test_batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_data, test_batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, person_id


def log_values(writer, step, loss, accuracy, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/accuracy", accuracy, step)


'''
Input arguments
  batch_size: Size of a mini-batch
  device: GPU where you want to train your network
  weight_decay: Weight decay co-efficient for regularization of weights
  momentum: Momentum for SGD optimizer
  epochs: Number of epochs for training the network
'''


def main(batch_size=128,
         device='cuda:0',
         learning_rate=1e-3,
         weight_decay=5e-4,
         momentum=0.9,
         epochs=20):

    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="runs/exp1")

    train_loader, val_loader, test_loader, person_id = get_data(batch_size)

    # Inserire Num persone
    net = CanaleCinque(
        train_loader.dataset.dataset.id_to_internal_id[person_id] + 1).to(device)

    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

    print('Before training:')
    train_loss, train_accuracy = test(net, train_loader)
    # val_loss, val_accuracy = test(net, val_loader)
    # test_loss, test_accuracy = test(net, test_loader)

    log_values(writer, -1, train_loss, train_accuracy, "Train")
    # log_values(writer, -1, val_loss, val_accuracy, "Validation")
    # log_values(writer, -1, test_loss, test_accuracy, "Test")

    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(
        train_loss, train_accuracy))
    # print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(
    #     val_loss, val_accuracy))
    # print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(
    #     test_loss, test_accuracy))
    print('-----------------------------------------------------')

    for e in range(epochs):
        train_loss, train_accuracy = train(
            net, train_loader, optimizer)
        # val_loss, val_accuracy = test(net, val_loader)

        # log_values(writer, e, val_loss, val_accuracy, "Validation")

        print('Epoch: {:d}'.format(e+1))
        print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(
            train_loss, train_accuracy))
        # print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(
        #     val_loss, val_accuracy))
        print('-----------------------------------------------------')

    print('After training:')
    train_loss, train_accuracy = test(net, train_loader)
    # val_loss, val_accuracy = test(net, val_loader)
    # test_loss, test_accuracy = test(net, test_loader)

    log_values(writer, epochs, train_loss, train_accuracy, "Train")
    # log_values(writer, epochs, val_loss, val_accuracy, "Validation")
    # log_values(writer, epochs, test_loss, test_accuracy, "Test")

    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(
        train_loss, train_accuracy))
    # print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(
    #     val_loss, val_accuracy))
    # print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(
    #     test_loss, test_accuracy))
    print('-----------------------------------------------------')

    writer.close()


if __name__ == '__main__':
    main()
