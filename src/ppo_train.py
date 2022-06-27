import numpy as np
import torch
from torch.utils.data import DataLoader
import pm_net
import random
import os
import pickle
from tqdm import tqdm


def train_pm(args, iteration):
    print('Starting model training....')
    dataset = {'data': []}
    data_path = f"./data/ppo_data/game_data/iter_{iteration}/"

    for idx,file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            game_data = pickle.load(fo, encoding='bytes')
            dataset['expansions'] = game_data['expansions']
            dataset['data'].extend(game_data['data'])

    train_dataset = pm_net.get_dataset(args.pm_id, dataset['data'], dataset['expansions'], ppo=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model_base_path = f'./data/ppo_data/pm_model_data/{args.pm_id}/'

    pm = pm_net.get_pm(args.pm_id)
    pm_old = pm_net.get_pm(args.pm_id)

    if iteration == 0:
        model_path = os.path.join(model_base_path, f'{args.pm_id}_iter_0.pth.tar')
        pm.load_state_dict(torch.load(model_path)['state_dict'])
        pm_old.load_state_dict(torch.load(model_path)['state_dict'])
    else:
        model_path_old = os.path.join(model_base_path, f'{args.pm_id}_iter_{iteration-1}.pth.tar')
        model_path = os.path.join(model_base_path, f'{args.pm_id}_iter_{iteration}.pth.tar')
        pm_old.load_state_dict(torch.load(model_path_old)['state_dict'])
        pm.load_state_dict(torch.load(model_path)['state_dict'])

    if torch.cuda.is_available():
        pm.cuda()
        pm_old.cuda()

    pm.train()

    optimizer = torch.optim.Adam(pm.parameters(), lr=args.lr)
    loss_fn = pm_net.PPO_Loss()

    # Pass data one Epoch
    for i, data in enumerate(tqdm(train_loader)):
        # Every data instance is an input + label pair
        inputs, labels, mask = data
        labels = torch.tensor([[l]*7 for l in labels], dtype=torch.float32)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            mask = mask.cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = pm(inputs)
        outputs_old = pm_old(inputs)

        # Compute the loss and its gradient
        loss = loss_fn(outputs, outputs_old, labels, mask, args.clip_epsilon)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

    path = os.path.join(model_base_path, f'{args.pm_id}_iter_{iteration+1}.pth.tar')
    torch.save({'state_dict': pm.state_dict(), 'pm_id': args.pm_id}, path)
