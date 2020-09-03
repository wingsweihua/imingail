import torch
import torch.nn as nn
import os
from utils.traj_utils import *


class FCNetwork(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size):

        super(FCNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):

        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x


def model_train(args, dataloaders, optimizer, net, criterion):

    # training

    print('Training interpolation model...')

    list_train_loss = []

    for epoch in range(args.num_epochs):

        running_loss = 0.0

        for i, data in enumerate(dataloaders["train"], 0):

            x_batch, y_batch = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x_batch.float())
            if isinstance(criterion, list):
                for index, _loss in enumerate(criterion):
                    if index == 0:
                        loss = _loss(outputs, y_batch.float())
                    else:
                        loss += 0.0*_loss(outputs, y_batch.float())
            else:
                loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss/100))
                running_loss = 0.0

                list_train_loss.append([epoch + 1, i + 1, running_loss/100])

    # save model
    if not os.path.exists(os.path.join(args.data_dir, args.memo)):
        os.mkdir(os.path.join(args.data_dir, args.memo))
    if not os.path.exists(os.path.join(args.data_dir, args.memo, args.scenario)):
        os.mkdir(os.path.join(args.data_dir, args.memo, args.scenario))
    torch.save(net, os.path.abspath(os.path.join(args.data_dir, args.memo, args.scenario, args.model_name)))

    df_train_loss = pd.DataFrame(list_train_loss, columns=["epoch", "i", "loss"])
    df_train_loss.to_csv(os.path.join(args.data_dir, args.memo, args.scenario,
                                      '{0}_training_loss.csv'.format(args.experiment_name)), index=False)

    print('Finished Training')

