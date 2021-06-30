import torch
import pandas as pd


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

            loss = net.get_test_loss(outputs, targets)
            cumulative_loss += loss.item()

            predicteds = {}
            for key, value in outputs.items():
                _, predicteds[key] = value.max(1)

            accuracy = net.get_test_accuracy(predicteds, targets)
            cumulative_accuracy += accuracy

    return cumulative_loss/samples, cumulative_accuracy/samples*100


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

            for name in img_names['file_name']:
                if "id" in total_predicteds:
                    total_predicteds["id"].append(name)
                else:
                    total_predicteds["id"] = [name]

            batch_predicted = unpack_annotation(batch_predicted)

            for key, value in batch_predicted.items():
                for v in value:
                    if key in total_predicteds:
                        total_predicteds[key].append(v)
                    else:
                        total_predicteds[key] = [v]

    df = pd.DataFrame(total_predicteds)
    df.to_csv("annotations_train.csv", index=False)

    return True


def unpack_annotation(annotation):
    dictionary = {}

    dictionary["age"] = annotation["age"].cpu().numpy() + 1
    dictionary["backpack"] = annotation["carrying_backpack"].cpu().numpy() + 1
    dictionary["bag"] = annotation["carrying_bag"].cpu().numpy() + 1
    dictionary["handbag"] = annotation["carrying_handbag"].cpu().numpy() + 1
    dictionary["clothes"] = annotation["type_lower_body_clothing"].cpu().numpy() + 1
    dictionary["down"] = annotation["length_lower_body_clothing"].cpu().numpy() + 1
    dictionary["up"] = annotation["sleeve_length"].cpu().numpy() + 1
    dictionary["hair"] = annotation["hair_length"].cpu().numpy() + 1
    dictionary["hat"] = annotation["wearing_hat"].cpu().numpy() + 1
    dictionary["gender"] = annotation["gender"].cpu().numpy() + 1

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
        for i in range(9):
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

    for k in annotation["color_lower_body_clothing"]:
        for i in range(10):
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
