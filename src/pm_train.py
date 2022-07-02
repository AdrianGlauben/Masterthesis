import numpy as np
import torch
from torch.utils.data import DataLoader
import pm_net
import random
import os
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn.functional as F

dataset = {'data': []}
data_path = "./data/pm_data/game_data/"

for idx,file in enumerate(os.listdir(data_path)):
    filename = os.path.join(data_path,file)
    with open(filename, 'rb') as fo:
        game_data = pickle.load(fo, encoding='bytes')
        dataset['expansions'] = game_data['expansions']
        dataset['data'].extend(game_data['data'])

dataset['data'] = random.sample(dataset['data'], k=1250000)
split = int(0.8*len(dataset['data']))
train_data = dataset['data'][:split]
validation_data = dataset['data'][split:]


################################################
#### Chose one by commenting the other out ####
################################################

####               Simple PM                ####
# train_set = pm_net.SPMDataset_QVar(train_data)
# validation_set = pm_net.SPMDataset_QVar(validation_data)
# model = pm_net.SimplePM_QVar()

####                Conv PM                 ####
train_set = pm_net.ConvPMDataset_All(train_data, dataset['expansions'])
validation_set = pm_net.ConvPMDataset_All(validation_data, dataset['expansions'])
model = pm_net.ConvPM_All()

#################################################

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=32, shuffle=True)

if torch.cuda.is_available():
    model.cuda()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('./data/tb_logs/{}_trainer_{}'.format(model.id, timestamp))
epoch_number = 0

EPOCHS = 30

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    vacc = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        if torch.cuda.is_available():
            vinputs = vinputs.cuda()
            vlabels = vlabels.cuda()
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss.item()
        for pred, target in zip(voutputs, vlabels):
            if torch.argmax(pred) == target:
                vacc += 1


    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {} // ACC valid {}'.format(avg_loss, avg_vloss, vacc/len(validation_set)))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = './data/pm_data/models/{}_{}'.format(model.id, epoch_number)
        torch.save({'state_dict': model.state_dict(), 'pm_id':'SPM_base'}, model_path)

    epoch_number += 1
