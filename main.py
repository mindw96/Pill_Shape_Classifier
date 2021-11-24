from model import Shape_Effi_B7, Shape_Effi_B0, Shape_ResNet152, Shape_ResNet18
from utils import train_model, load_data, draw_graph
import torch
import torchvision


def main(learning_rate=0.1, batch_size=64, epoch=100):
    dataloader, class_num = load_data(batch_size=batch_size)

    model = Shape_Effi_B0(class_num=class_num)
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.fc.weight.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    lmbda = lambda epoch: 0.98739

    exp_lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer_adam, lr_lambda=lmbda)

    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(model, criterion,
                                                                                          optimizer_adam,
                                                                                          exp_lr_scheduler,
                                                                                          num_epochs=epoch,
                                                                                          dataloader=dataloader)

    draw_graph(best_idx, train_acc, train_loss, valid_acc, valid_loss)


if __name__ == '__main__':
    learning_rate = 0.0000001
    batch_size = 128
    epoch = 50
    main(learning_rate=learning_rate, batch_size=batch_size, epoch=epoch)
