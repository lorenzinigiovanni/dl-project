import torch
import torchvision
from torchvision import transforms as T
import torch.nn.functional as F
import torch.nn as nn
import os
import pandas as pd
from Market1501Dataset import Market1501Dataset

# Reti presidenziali
from ReteQuattro import ReteQuattro
from CanaleCinque import CanaleCinque


def get_optimizer(net, lr, wd, momentum):
    # optimizer = torch.optim.SGD(
    #     net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    return optimizer


def unpack_annotation(annotation):
    dictionary = {}

    dictionary["age"] = annotation["age"].cpu().numpy() + 1
    dictionary["backpack"] = annotation["carrying_backpack"].cpu().numpy() + 1
    dictionary["bag"] = annotation["carrying_bag"].cpu().numpy() + 1
    dictionary["handbag"] = annotation["carrying_handbag"].cpu().numpy() + 1
    dictionary["clothes"] = annotation["type_lower_body_clothing"].cpu().numpy() + 1
    dictionary["down"] = annotation["length_lower_body_clothing"].cpu().numpy() + 1
    dictionary["up"] = annotation["sleeve_lenght"].cpu().numpy() + 1
    dictionary["hair"] = annotation["hair_length"].cpu().numpy() + 1
    dictionary["hat"] = annotation["wearing_hat"].cpu().numpy() + 1
    dictionary["gender"] = annotation["gender"].cpu().numpy() + 1

    # upblack,upwhite,upred,uppurple,upyellow,upgray,upblue,upgreen
    up_color_dict = {
        0: "upmulticolor",
        1: "upblack",
        2: "upwhite",
        3: "upred",
        4: "uppurple",
        5: "upyellow",
        6: "upgray",
        7: "upblue",
        8: "upgreen"
    }

    down_color_dict = {
        0: "downmulticolor",
        1: "downblack",
        2: "downwhite",
        3: "downpink",
        4: "downpurple",
        5: "downyellow",
        6: "downgray",
        7: "downblue",
        8: "downgreen",
        9: "downbrown"
    }

    for k in annotation["color_upper_body_clothing"]:
        for i in range(9):  # upper body color
            if i == k:
                if up_color_dict[i] in dictionary:
                    dictionary[up_color_dict[i]].append(2)
                else:
                    dictionary[up_color_dict[i]] = [2]
            else:
                if up_color_dict[i] in dictionary:
                    dictionary[up_color_dict[i]].append(1)
                else:
                    dictionary[up_color_dict[i]] = [1]

    for k in annotation["color_upper_body_clothing"]:
        for i in range(10):  # lower body color
            if i == k:
                if down_color_dict[i] in dictionary:
                    dictionary[down_color_dict[i]].append(2)
                else:
                    dictionary[down_color_dict[i]] = [2]
            else:
                if down_color_dict[i] in dictionary:
                    dictionary[down_color_dict[i]].append(1)
                else:
                    dictionary[down_color_dict[i]] = [1]

    return dictionary


def annotate_csv(net, data_loader, device='cuda:0'):
    total_predicteds = {}

    net.eval()
    with torch.no_grad():
        for (inputs, img_names) in data_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)

            batch_predicted = {}
            for key, value in outputs.items():
                _, batch_predicted[key] = value.max(1)

            for name in img_names:
                if "id" in total_predicteds:
                    total_predicteds["id"].append(name)
                else:
                    total_predicteds["id"] = [name]

            batch_predicted = unpack_annotation(batch_predicted)

            for key, value in batch_predicted.items():
                for v in value:
                    # if isinstance(v, torch.Tensor):
                    #     v = v.item()
                    if key in total_predicteds:
                        total_predicteds[key].append(v)
                    else:
                        total_predicteds[key] = [v]

    df = pd.DataFrame(total_predicteds)
    df.to_csv("annotations_train.csv", index=False)

    return True


def train(net, data_loader, optimizer, device='cuda:0'):
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    net.train()
    for (inputs, targets) in data_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)

        samples += inputs.shape[0]

        loss, losses = net.get_loss(outputs, targets)
        cumulative_loss += loss.item()

        predicteds = {}
        for key, value in outputs.items():
            _, predicteds[key] = value.max(1)

        accuracy, accuracies = net.get_accuracy(predicteds, targets)
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

            loss, losses = net.get_loss(outputs, targets)
            cumulative_loss += loss.item()

            predicteds = {}
            for key, value in outputs.items():
                _, predicteds[key] = value.max(1)

            accuracy, accuracies = net.get_accuracy(predicteds, targets)
            cumulative_accuracy += accuracy

    return cumulative_loss/samples, cumulative_accuracy/samples*100


def create_translation_transform():
    '''
    Creates a transformation that pads the original so that the original portion
    will not be always centered
    '''
    def padding(x):
        pad_size = 28
        left_padding = torch.randint(low=0, high=pad_size, size=(1,))
        top_padding = torch.randint(low=0, high=pad_size, size=(1,))
        return F.pad(x, (left_padding,
                         pad_size - left_padding,
                         top_padding,
                         pad_size - top_padding), "constant", 0)

    translation_transform = list()
    # translation_transform.append(T.ToTensor())
    # Normalizes the Tensors between [-1, 1]
    translation_transform.append(T.ConvertImageDtype(torch.float))
    translation_transform.append(T.Normalize(mean=[0.5], std=[0.5]))
    translation_transform.append(T.Lambda(lambda x: padding(x)))
    translation_transform = T.Compose(translation_transform)

    return translation_transform


def get_data(batch_size, test_batch_size=256, translate=False):

    if not translate:
        # Prepare data transformations and then combine them sequentially
        transform = list()
        # converts Numpy to Pytorch Tensor
        # transform.append(T.ToTensor())
        # transform.append(T.RandomCrop(32, padding=4))
        # transform.append(T.RandomHorizontalFlip())
        transform.append(T.ConvertImageDtype(torch.float))

        transform.append(T.Normalize(
            # Roba trovata da una Github di un ciro che ha fatto cose sul Market1501
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        )

        # transform.append(T.Normalize(mean=[0.48], std=[0.25]))      # Normalizes the Tensors between [-1, 1]
        # transform.append(T.Normalize(
        #     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        # transform.append(T.Lambda(lambda x: F.pad(x, (14, 14, 14, 14), "constant", 0)))
        # transform.append(T.ColorJitter(hue=0.1)
        # Composes the above transformations into one.
        transform = T.Compose(transform)
    else:
        # Applies random translations to images
        transform = create_translation_transform()

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
    while (person_id == full_training_data.dict[training_samples][1]):
        training_samples += 1

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

    return train_loader, val_loader, test_loader


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
         epochs=3):

    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="runs/exp1")

    train_loader, val_loader, test_loader = get_data(batch_size)

    net = ReteQuattro().to(device)

    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

    print('Before training:')
    train_loss, train_accuracy = test(net, train_loader)
    val_loss, val_accuracy = test(net, val_loader)
    # test_loss, test_accuracy = test(net, test_loader)

    log_values(writer, -1, train_loss, train_accuracy, "Train")
    log_values(writer, -1, val_loss, val_accuracy, "Validation")
    # log_values(writer, -1, test_loss, test_accuracy, "Test")

    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(
        train_loss, train_accuracy))
    print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(
        val_loss, val_accuracy))
    # print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(
    #     test_loss, test_accuracy))
    print('-----------------------------------------------------')

    for e in range(epochs):
        train_loss, train_accuracy = train(
            net, train_loader, optimizer)
        val_loss, val_accuracy = test(net, val_loader)

        log_values(writer, e, val_loss, val_accuracy, "Validation")

        print('Epoch: {:d}'.format(e+1))
        print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(
            train_loss, train_accuracy))
        print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(
            val_loss, val_accuracy))
        print('-----------------------------------------------------')

    print('After training:')
    train_loss, train_accuracy = test(net, train_loader)
    val_loss, val_accuracy = test(net, val_loader)
    # test_loss, test_accuracy = test(net, test_loader)

    log_values(writer, epochs, train_loss, train_accuracy, "Train")
    log_values(writer, epochs, val_loss, val_accuracy, "Validation")
    # log_values(writer, epochs, test_loss, test_accuracy, "Test")

    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(
        train_loss, train_accuracy))
    print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(
        val_loss, val_accuracy))
    # print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(
    #     test_loss, test_accuracy))
    print('-----------------------------------------------------')

    writer.close()

    annotate_csv(net, test_loader)


if __name__ == '__main__':
    main()
