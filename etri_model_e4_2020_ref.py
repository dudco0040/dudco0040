import torch
import pandas
import numpy as np
from utils import *
from models import *
import copy
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support as score

import torch.nn as nn
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(1024, 1000) #1024-15*15 #4096-w/meta
        self.fc2 = nn.Linear(1000, 4)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


USER_TRAIN= [
    "user01",
    "user02",
    "user03",
    "user04",
    "user05",
    "user08",
    "user09",
    "user10",
    "user11",
    "user12",
    "user21",
    "user22",
    "user23",
    "user24",
    "user25",
    "user26",
    "user27",
    "user28",
    "user29",
    "user30",
    "user006",
    "user008",
]

class SensorTrainDataset(Dataset):
    """ Sensor dataset for training."""
    # Initialize data (pre-processing)
    def __init__(self):
        self.len = cv_train_dataset.shape[0]
        self.x_data = torch.from_numpy(cv_train_dataset).float()
        self.y_data = torch.from_numpy(cv_train_labels)
        print(self.x_data.shape)
        print(self.y_data.shape)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def train_test(model, criterion, optimizer, scheduler, trainloader, testloader, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    #trainloader, testloader, train_len, test_len = get_data_loaders()

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for train in [True]:
            if train:
                model.train()  # Set model to training mode
                dataloader = trainloader
                #data_len = train_len
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = testloader
                #data_len = test_len

            running_loss = 0.0
            running_corrects = 0
            total = 0
            correct = 0
            labels_arr = [] #labels
            output_arr = []
            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                #inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(train == True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.long())

                    # backward + optimize only if in training phase
                    if train:
                        loss.backward()
                        optimizer.step()
                #print(preds)
                #print(labels.data)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                preds_ = preds.cpu()
                labels_ = labels.cpu()
                preds_ = preds_.numpy()
                labels_ = labels_.numpy()
                output_arr = np.append(output_arr, preds_)
                labels_arr = np.append(labels_arr, labels_)

            #if train:
            #    scheduler.step()

            epoch_loss = running_loss / total #dataset_sizes[phase]
            epoch_acc = running_corrects.double() / total #dataset_sizes[phase]

            precision, recall, fscore, support = score(labels_arr, output_arr, average='weighted')

            print('Epoch {}/{} Loss: {:.4f} Acc: {:.4f} Prec: {:.4f} Recall: {:.4f} fscore: {:.4f}'.format(
                epoch, num_epochs - 1, epoch_loss, epoch_acc, precision, recall, fscore))#

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fpath = '../../../data_mongodb/2020_e4/' #path to read dataset
    spath = fpath + 'models' #path to save HAR model

    # Make save directory
    if not os.path.exists(spath):
        os.makedirs(spath)
    if not os.path.isdir(spath):
        raise Exception('%s is not a dir' % spath)

    for user_train in USER_TRAIN:
        print(user_train)
        fdata = fpath + str(user_train) + '_e4.npy'
        flabel = fpath + str(user_train) + '_label.npy'

        train_dataset = np.load(fdata)
        train_label = np.load(flabel)

        print(len(train_dataset))
        idx_cnt = 0
        for idx in range(len(train_dataset)):
            one_x = train_dataset[idx][:-5, 0]
            one_y = train_dataset[idx][:-5, 1]
            one_z = train_dataset[idx][:-5, 2]
            new_x = np.reshape(one_x, (5, 15))  # reshape to 5 by 15 matrix
            new_y = np.reshape(one_y, (5, 15))
            new_z = np.reshape(one_z, (5, 15))
            fin_d = np.concatenate((new_x, new_y, new_z), axis=0)  # concatenate 2D arrays
            fin_d = np.reshape(fin_d, (1, 15, 15))
            one_l = train_label[idx, 0]

            if idx_cnt == 0:
                cv_train_dataset = fin_d
                cv_train_labels = one_l
            elif idx_cnt == 1:  # second iteration
                cv_train_dataset = np.array([cv_train_dataset, fin_d])  # stack 3D arrays to 4D
                cv_train_labels = np.append(cv_train_labels, one_l)
            else:
                fin_d = np.reshape(fin_d, (1, 1, 15, 15))
                cv_train_dataset = np.concatenate((cv_train_dataset, fin_d))
                cv_train_labels = np.append(cv_train_labels, one_l)
            idx_cnt += 1

        dset = SensorTrainDataset()
        train_loader = DataLoader(dset, batch_size=128, shuffle=True)

        ##################################################################################################################

        model_org = ConvNet()
        model_org = model_org.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD([v for v in model_org.parameters() if v.requires_grad], lr=0.0001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        model_org = train_test(model_org, criterion, optimizer, scheduler, train_loader, train_loader, num_epochs=300)
        #print(model_org)

        # Save model
        torch.save(model_org.state_dict(), os.path.join(spath, str(user_train) + '_model.pt'))


