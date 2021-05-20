import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class Market1501Dataset(Dataset):
    def __init__(self, img_dir, annotations_file=None, transform=None, device="cuda:0"):
        self.annotations_file = annotations_file
        if(annotations_file != None):
            self.img_labels = pd.read_csv(annotations_file, sep=",", header=0)
        self.img_dir = img_dir
        self.transform = transform
        self.device = device
        self.dict = {}
        self.id_to_internal_id = {}
        internal_id = 0

        for i, img_name in enumerate(os.listdir(img_dir)):
            if(annotations_file != None):
                id = int(img_name.split("_")[0])
                self.dict[i] = (img_name, id)
                if id not in self.id_to_internal_id:
                    self.id_to_internal_id[id] = internal_id
                    internal_id += 1
            else:
                self.dict[i] = (img_name, int(img_name.split(".")[0]))

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.dict[idx][0])
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        if(self.annotations_file != None):
            labels = self.img_labels.loc[self.img_labels['id']
                                         == self.dict[idx][1]]
            labels = labels.to_numpy(dtype="long")[0]
            labels = labels - 1

            j = 0
            for i in range(11, 19):
                if(labels[i] == 1):
                    j = i-10
                    break

            k = 0
            for i in range(19, 28):
                if(labels[i] == 1):
                    k = i-18
                    break

            labels = labels[1:11]

            labels = torch.from_numpy(labels).long().to(self.device)

            labels = {
                'age': labels[0],
                'carrying_backpack': labels[1],
                'carrying_bag': labels[2],
                'carrying_handbag': labels[3],
                'type_lower_body_clothing': labels[4],
                'length_lower_body_clothing': labels[5],
                'sleeve_lenght': labels[6],
                'hair_length': labels[7],
                'wearing_hat': labels[8],
                'gender': labels[9],
                'color_upper_body_clothing': torch.tensor(j, dtype=torch.long).to(self.device),
                'color_lower_body_clothing': torch.tensor(k, dtype=torch.long).to(self.device),
                'internal_id': self.id_to_internal_id[self.dict[idx][1]]
            }

            return (image, labels)

        else:
            return (image, self.dict[idx][0])
