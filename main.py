import os
from Market1501Dataset import Market1501Dataset

def main():
    train = Market1501Dataset(os.path.join(
        "dataset", "annotations_train.csv"), os.path.join("dataset", "train"))
    for i in range(0, 10):
        print(train[i])


if __name__ == "__main__":
    main()
