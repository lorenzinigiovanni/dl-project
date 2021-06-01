import torch
from torchvision import transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import pandas as pd
from Market1501Dataset import Market1501Dataset
import numpy as np
import random

from random_erasing import RandomErasing

# Reti presidenziali
from ItaliaUno import ItaliaUno


def get_optimizer(net, lr, wd, momentum):
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


def classification_test(net, data_loader, device='cuda:0'):
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    net.eval()
    with torch.no_grad():
        for (inputs, targets) in data_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)

            samples += inputs.shape[0]

            loss = net.get_test_loss(outputs, targets)
            cumulative_loss += loss.item()  # Il .item() estrae uno scalare da un tensore

            predicteds = {}
            for key, value in outputs.items():
                _, predicteds[key] = value.max(1)

            accuracy = net.get_test_accuracy(predicteds, targets)
            cumulative_accuracy += accuracy

    return cumulative_loss/samples, cumulative_accuracy/samples*100


def get_data(batch_size, test_batch_size=256):

    transform = list()

    transform.append(T.ConvertImageDtype(torch.float))
    transform.append(T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    )
    transform.append(T.Resize((256, 128)))

    training_transform = list()

    training_transform.append(T.RandomCrop((256, 128), 10))
    training_transform.append(T.RandomHorizontalFlip())
    training_transform.append(RandomErasing())

    # Load data
    full_training_data = Market1501Dataset(
        os.path.join("dataset", "train"),
        os.path.join("dataset", "annotations_train.csv"),
        transform=T.Compose(transform + training_transform)
    )
    test_data = Market1501Dataset(
        os.path.join("dataset", "test"),
        transform=T.Compose(transform)
    )
    query_data = Market1501Dataset(
        os.path.join("dataset", "queries"),
        transform=T.Compose(transform)
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
    query_loader = torch.utils.data.DataLoader(
        query_data, test_batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, query_loader, person_id


def get_vectors(net, data_loader, device='cuda:0'):
    outputs = []
    names = []

    net.eval()
    with torch.no_grad():
        for (inputs, targets) in data_loader:
            inputs = inputs.to(device)
            output = net(inputs)

            for i in range(output['age'].shape[0]):
                big_vector = []

                for k, v in output.items():
                    if k == 'internal_id':
                        continue

                    big_vector.append(v[i])

                outputs.append(torch.cat(big_vector))

            names += targets['file_name']

    return torch.stack(outputs), names


def query(query, values, names, th=30):
    outputs = []

    for i, x in enumerate(values):
        mse = (np.square(query - x)).mean()
        if mse < th:
            outputs.append((names[i], mse))

    outputs.sort(key=lambda tup: tup[1])

    return outputs


def predict_id(values, names):
    values = values.cpu().detach().numpy()

    results = {}

    random.seed("forza_napoli")

    for _ in range(100):
        i = random.randint(0, len(values) - 1)
        y = query(values[i], values, names)
        results[names[i]] = y

    return results


def evaluate(net, data_loader):
    values, names = get_vectors(net, data_loader)

    results = predict_id(values, names)

    aPiacere = {}
    for k, v in results.items():
        aPiacere[k] = [x for x, _ in v]

    ground_truth = {}
    for k in results:
        ground_truth[k] = set(
            [
                name for name in names
                if name.split('_')[0] == k.split('_')[0]
            ]
        )

    from evaluator import Evaluator

    mAP = Evaluator.evaluate_map(aPiacere, ground_truth)

    return mAP


def answer_query(net, query_data_loader, test_data_loader):
    query_values, query_names = get_vectors(net, query_data_loader)
    test_values, test_names = get_vectors(net, test_data_loader)

    test_values = test_values.cpu().detach().numpy()

    results = {}

    for i in range(50):  # range(len(query_names)):
        y = query(
            query_values.cpu().detach().numpy()[i],
            test_values,
            test_names,
            th=5
        )
        results[query_names[i]] = [x for x, _ in y]

    return results


def write_answers_txt(dictionary):
    with open('reid_test.txt', 'w') as file:
        for k, values in dictionary.items():
            print(k + ':', end='', file=file)
            stringa = ''
            for v in values:
                stringa += ' ' + v + ','
            print(stringa[:-1], end='', file=file)
            print('', file=file)


def log_values(writer, step, loss, accuracy, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/accuracy", accuracy, step)


def main(batch_size=128,
         device='cuda:0',
         learning_rate=1e-3,
         weight_decay=5e-4,
         momentum=0.9,
         epochs=30,
         is_training=False):

    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

    from torch.utils.tensorboard import SummaryWriter

    name = "runs/exp" + str(len(os.listdir("runs"))+1)
    os.makedirs(name)

    writer = SummaryWriter(log_dir=name)

    train_loader, val_loader, test_loader, query_data, person_id = get_data(
        batch_size)

    # Inserire Num persone
    net = ItaliaUno(
        train_loader.dataset.dataset.id_to_internal_id[person_id] + 1).to(device)

    net.load_state_dict(model_zoo.load_url(
        'https://download.pytorch.org/models/resnet50-19c8e357.pth'), strict=False)

    if is_training:
        optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

        print('Before training:')
        train_loss, train_accuracy = test(net, train_loader)
        val_loss, val_accuracy = classification_test(net, val_loader)

        log_values(writer, -1, train_loss, train_accuracy, "Train")
        log_values(writer, -1, val_loss, val_accuracy, "Validation")
        writer.add_scalar("mAP", 0, -1)

        print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(
            train_loss, train_accuracy))
        print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(
            val_loss, val_accuracy))
        print('-----------------------------------------------------')

        for e in range(epochs):
            train_loss, train_accuracy = train(net, train_loader, optimizer)
            val_loss, val_accuracy = classification_test(net, val_loader)
            mAP = evaluate(net, val_loader)

            log_values(writer, e, train_loss, train_accuracy, "Train")
            log_values(writer, e, val_loss, val_accuracy, "Validation")
            writer.add_scalar("mAP", mAP, e)

            print('Epoch: {:d}'.format(e+1))
            print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(
                train_loss, train_accuracy))
            print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(
                val_loss, val_accuracy))
            print('\t mAP: ' + str(mAP))
            print('-----------------------------------------------------')

        print('After training:')
        train_loss, train_accuracy = test(net, train_loader)
        val_loss, val_accuracy = classification_test(net, val_loader)
        mAP = evaluate(net, val_loader)

        log_values(writer, epochs, train_loss, train_accuracy, "Train")
        log_values(writer, epochs, val_loss, val_accuracy, "Validation")
        writer.add_scalar("mAP", mAP, epochs)

        print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(
            train_loss, train_accuracy))
        print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(
            val_loss, val_accuracy))
        print('\t mAP: ' + str(mAP))
        print('-----------------------------------------------------')

        writer.close()

        with open('italia1_net.pt', 'wb') as f:
            torch.save(net, f)

    else:  # Evaluate the model
        with open('italia1_net.pt', 'rb') as f:
            model = torch.load(f)

            mAP = evaluate(model, val_loader)
            print('mAP: ' + str(mAP))

            val_loss, val_accuracy = classification_test(model, val_loader)
            print('Validation loss {:.5f}, Validation accuracy {:.2f}'.format(
                val_loss, val_accuracy))

            answers = answer_query(model, query_data, test_loader)
            write_answers_txt(answers)


if __name__ == '__main__':
    main()