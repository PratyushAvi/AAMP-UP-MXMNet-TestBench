from __future__ import division
from __future__ import print_function

import os.path as osp
import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from model import MXMNet, Config
from utils import EMA
from qm9_dataset import QM9

import matplotlib.pyplot as plt
import time
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
parser.add_argument('--seed', type=int, default=920, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay value.')
parser.add_argument('--n_layer', type=int, default=6, help='Number of hidden layers.')
parser.add_argument('--dim', type=int, default=128, help='Size of input hidden units.')
parser.add_argument('--dataset', type=str, default="QM9", help='Dataset')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--target', type=int, default="7", help='Index of target (0~12) for prediction')
parser.add_argument('--cutoff', type=float, default=5.0, help='Distance cutoff used in the global layer')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

target = args.target
if target in [7, 8, 9, 10]:
    target = target + 5
set_seed(args.seed)

targets = ['mu (D)', 'a (a^3_0)', 'e_HOMO (eV)', 'e_LUMO (eV)', 'delta e (eV)', 'R^2 (a^2_0)', 'ZPVE (eV)', 'U_0 (eV)', 'U (eV)', 'H (eV)', 'G (eV)', 'c_v (cal/mol.K)',]

def test(loader):
    error = 0
    ema.assign(model)

    for data in loader:
        data = data.to(device)
        output = model(data)
        error += (output - data.y).abs().sum().item()
    ema.resume(model)
    return error / len(loader.dataset)

class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, target]
        return data

#Download and preprocess dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'QM9')
dataset = QM9(path, transform=MyTransform()).shuffle()
print('# of graphs:', len(dataset))

# Split dataset
# train_dataset = dataset[:110000]
# val_dataset = dataset[110000:120000]
# test_dataset = dataset[120000:]

SET_SCALE = float(input("Enter Set Scale (0 - 1): "))
SET_RUNTIME = float(input("Enter run time in minutes: "))

TRAIN_START = int(110000 * SET_SCALE)
VAL_END = TRAIN_START + int(10000 * SET_SCALE)
TEST_END = VAL_END + int(10000 * SET_SCALE)

train_dataset = dataset[:TRAIN_START]
val_dataset = dataset[TRAIN_START:VAL_END]
test_dataset = dataset[VAL_END:TEST_END]

print("Training Data:", TRAIN_START)
print("Valid Data:", VAL_END-TRAIN_START)
print("Test Data:", TEST_END-VAL_END)

#f = open('../notes/QM9_MXMNet_latest_run.txt', 'w')

#Load dataset
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=args.seed)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

print('Loaded the QM9 dataset. Target property: ', targets[args.target])
# print('Loaded the QM9 dataset. Target property: ', 'HoF')

# Load model
config = Config(dim=args.dim, n_layer=args.n_layer, cutoff=args.cutoff)

model = MXMNet(config).to(device)
print('Loaded the MXMNet.')

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

ema = EMA(model, decay=0.999)

print('===================================================================================')
print('                                Start training:')
print('===================================================================================')

best_epoch = None
best_val_loss = None

train_MAEs = []
valid_MAEs = []
test_MAEs = []

my_epochs = args.epochs
epoch = 0

while epoch < my_epochs:
    if epoch == 0:
        START_TIME = time.time()

    loss_all = 0
    step = 0
    model.train()

    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = F.l1_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)
        optimizer.step()
        
        curr_epoch = epoch + float(step) / (len(train_dataset) / args.batch_size)
        scheduler_warmup.step(curr_epoch)

        ema(model)
        step += 1

    train_loss = loss_all / len(train_loader.dataset)

    val_loss = test(val_loader)


    if best_val_loss is None or val_loss <= best_val_loss:
        test_loss = test(test_loader)
        best_epoch = epoch
        best_val_loss = val_loss
    
    train_MAEs.append(train_loss)
    valid_MAEs.append(val_loss)
    test_MAEs.append(test_loss)

    print('Epoch: {:03d}, Train MAE: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f}'.format(epoch+1, train_loss, val_loss, test_loss))
    
    # f.write('Epoch: {:03d}, Train MAE: {:.7f}, Validation MAE: {:.7f}, '
    #       'Test MAE: {:.7f}\n'.format(epoch+1, train_loss, val_loss, test_loss))

    if epoch == 0:
        END_TIME = time.time()

        if my_epochs * (END_TIME - START_TIME) / 60 > SET_RUNTIME:
            print("Decreased epochs from", my_epochs, "to ", end="")   
        elif my_epochs * (END_TIME - START_TIME) / 60 < SET_RUNTIME: 
            print("Increased epochs from", my_epochs, "to ", end="")
        else:
            print("Epochs not changed")
            continue
            
        my_epochs = int((SET_RUNTIME * 60) / (END_TIME - START_TIME))
        print(my_epochs)
    
    epoch = epoch + 1

print('===================================================================================')
print('Best Epoch:', best_epoch)
print('Best Test MAE:', test_loss)

# f.write('Best Epoch: {:d}\n'.format(best_epoch))
# f.write('Best Test MAE: {:.7f}\n'.format(test_loss))

# f.close()


e_x = list(range(my_epochs))

plt.figure()
plt.plot(e_x, train_MAEs, label = 'Training', color = 'blue')
plt.plot(e_x, valid_MAEs, label = 'Validation', color = 'green')
plt.plot(e_x, test_MAEs, label = 'Testing', color = 'red')
plt.legend()
plt.savefig('../notes/benchmarks/graphs/%s_%.2f_%d_%d_graph.png' % (targets[args.target], SET_SCALE * 100, my_epochs, SET_RUNTIME))

with open('../notes/benchmarks/csvs/%s_%.2f_%d_%d_graph_data.csv' % (targets[args.target], SET_SCALE * 100, my_epochs, SET_RUNTIME), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['train_MAEs', 'valid_MAEs', 'test_MAEs'])
    for i in range(my_epochs):
        writer.writerow([train_MAEs[i], valid_MAEs[i], test_MAEs[i]])

plt.show()