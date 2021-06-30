import torch

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
            cumulative_loss += loss.item()

            predicteds = {}
            for key, value in outputs.items():
                _, predicteds[key] = value.max(1)

            accuracy = net.get_accuracy(predicteds, targets)
            cumulative_accuracy += accuracy

    return cumulative_loss/samples, cumulative_accuracy/samples*100
