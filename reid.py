import torch
import random
import numpy as np
from evaluator import Evaluator


# get the mAP
def test(net, data_loader):
    values, names = get_vectors(net, data_loader)

    predictions = {}

    random.seed("forza_napoli")

    for _ in range(100):
        i = random.randint(0, len(values) - 1)
        y = query(values[i], values, names)
        predictions[names[i]] = y

    ground_truth = {}
    for k in predictions:
        ground_truth[k] = set([name for name in names if name.split('_')[0] == k.split('_')[0]])

    mAP = Evaluator.evaluate_map(predictions, ground_truth)

    return mAP


# accept a query dataset and a target dataset, return all the answered queries
def answer_query(net, query_data_loader, test_data_loader):
    query_values, query_names = get_vectors(net, query_data_loader)
    test_values, test_names = get_vectors(net, test_data_loader)

    test_values = test_values.cpu().detach().numpy()

    predictions = {}

    for i in range(50):  # range(len(query_names)):
        predictions[query_names[i]] = query(
            query_values.cpu().detach().numpy()[i],
            test_values,
            test_names,
            th=5
        )

    return predictions


# get an ordered list of file names from a query tensor
def query(query, values, names, th=30):
    predictions = []

    for i, x in enumerate(values):
        mse = (np.square(query - x)).mean()
        if mse < th:
            predictions.append((names[i], mse))

    predictions.sort(key=lambda tup: tup[1])

    return [x for x, _ in predictions]


# get network outputs and files names
def get_vectors(net, data_loader, device='cuda:0'):
    outputs = []
    names = []

    net.eval()
    with torch.no_grad():
        for (inputs, targets) in data_loader:
            output = net(inputs.to(device))
            outputs.append(output['pre_id'])
            names += targets['file_name']

    return torch.cat(outputs, dim=0), names


# store the answered queries in a txt file
def write_answers_txt(dictionary):
    with open('reid_test.txt', 'w') as file:
        for k, values in dictionary.items():
            print(k + ':', end='', file=file)
            stringa = ''
            for v in values:
                stringa += ' ' + v + ','
            print(stringa[:-1], end='', file=file)
            print('', file=file)
