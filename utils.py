import copy
import time
import matplotlib.pyplot as plt
import torch

from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def load_data(batch_size):
    data_path = './data/shape'

    shape_dataset = datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.Grayscale(3),
            transforms.RandomRotation((-90, 90)),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

    label_list = shape_dataset.classes

    train_idx, valid_idx = train_test_split(list(range(len(shape_dataset))), test_size=0.2, random_state=9608)

    dataset = {}
    dataset['train'] = Subset(shape_dataset, train_idx)
    dataset['valid'] = Subset(shape_dataset, valid_idx)

    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(dataset['train'],
                                                       batch_size=batch_size, shuffle=True,
                                                       num_workers=0)
    dataloader['valid'] = torch.utils.data.DataLoader(dataset['valid'],
                                                       batch_size=batch_size, shuffle=False,
                                                       num_workers=0)

    class_num = len(label_list)

    return dataloader, class_num


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, dataloader=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    since = time.time()
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('{} Loss: {:.2f} Acc: {:.2f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('==> best model saved - %d / %.2f' % (best_idx + 1, best_acc))

    time_elapsed = time.time() - since
    print('=' * 10)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx + 1, best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'shape_model.pt')
    print('model saved')

    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc


def draw_graph(best_idx, train_acc, train_loss, valid_acc, valid_loss):
    print('best model : %d - %1.f / %.1f' % (best_idx, valid_acc[best_idx], valid_loss[best_idx]))
    fig = plt.figure()
    ax1 = fig.subplots()

    ax1.plot(train_acc, 'b-', label='train_acc')
    ax1.plot(valid_acc, 'r-', label='valid_acc')
    plt.plot(best_idx, valid_acc[best_idx], 'ro')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('acc', color='k')
    ax1.tick_params('y', colors='k')

    ax2 = ax1.twinx()
    ax2.plot(train_loss, 'g-', label='train_loss')
    ax2.plot(valid_loss, 'k-', label='valid_loss')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    plt.ylim(-0.1, 10)
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params('y', colors='k')

    fig.legend()

    fig.tight_layout()
    plt.show()
