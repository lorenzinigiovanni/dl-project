from tensorboardX import SummaryWriter
import torch
import torch.utils.model_zoo as model_zoo
import os

from network import Network
import dataloader
import training
import classification
import reid


def log_values(writer, step, loss, accuracy, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/accuracy", accuracy, step)


def main(
    batch_size=128,
    test_batch_size=256,
    device='cuda:0',
    learning_rate=1e-3,
    weight_decay=5e-4,
    epochs=30,
    is_training=True
):

    print(torch.cuda.get_device_name(0))

    name = "runs/exp" + str(len(os.listdir("runs"))+1)
    os.makedirs(name)

    writer = SummaryWriter(log_dir=name)

    (
        train_loader,
        val_loader,
        test_loader,
        query_data,
        person_id
    ) = dataloader.get_data(
        batch_size,
        test_batch_size
    )

    # Inserire numero persone
    net = Network(train_loader.dataset.dataset.id_to_internal_id[person_id] + 1).to(device)

    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'), strict=False)

    if is_training:
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        print('Before training:')
        train_loss, train_accuracy = training.test(net, train_loader)
        val_loss, val_accuracy = classification.test(net, val_loader)
        mAP = reid.test(net, val_loader)

        log_values(writer, -1, train_loss, train_accuracy, "Train")
        log_values(writer, -1, val_loss, val_accuracy, "Validation")
        writer.add_scalar("mAP", mAP, -1)

        print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
        print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
        print('\t mAP: ' + str(mAP))
        print('-----------------------------------------------------')

        for e in range(epochs):
            train_loss, train_accuracy = training.train(net, train_loader, optimizer)
            val_loss, val_accuracy = classification.test(net, val_loader)
            mAP = reid.test(net, val_loader)

            log_values(writer, e, train_loss, train_accuracy, "Train")
            log_values(writer, e, val_loss, val_accuracy, "Validation")
            writer.add_scalar("mAP", mAP, e)

            print('Epoch: {:d}'.format(e+1))
            print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
            print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
            print('\t mAP: ' + str(mAP))
            print('-----------------------------------------------------')

        print('After training:')
        train_loss, train_accuracy = training.test(net, train_loader)
        val_loss, val_accuracy = classification.test(net, val_loader)
        mAP = reid.test(net, val_loader)

        log_values(writer, epochs, train_loss, train_accuracy, "Train")
        log_values(writer, epochs, val_loss, val_accuracy, "Validation")
        writer.add_scalar("mAP", mAP, epochs)

        print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
        print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
        print('\t mAP: ' + str(mAP))
        print('-----------------------------------------------------')

        writer.close()

        with open('network.pt', 'wb') as f:
            torch.save(net, f)

    else: 
        with open('network.pt', 'rb') as f:
            model = torch.load(f)

            mAP = reid.test(model, val_loader)
            print('mAP: ' + str(mAP))

            val_loss, val_accuracy = classification.test(model, val_loader)
            print('Validation loss {:.5f}, Validation accuracy {:.2f}'.format(
                val_loss, val_accuracy))

            answers = reid.answer_query(model, query_data, test_loader)
            reid.write_answers_txt(answers)

            classification.annotate_csv(net, test_loader)


if __name__ == '__main__':
    main()
