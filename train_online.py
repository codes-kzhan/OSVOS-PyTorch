# Package Includes
from __future__ import division

import os
import socket
import timeit
from datetime import datetime

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
from dataloaders import davis_2017 as db
from dataloaders import custom_transforms as tr
import scipy.misc as sm
import networks.vgg_osvos as vo
from layers.osvos_layers import class_balanced_cross_entropy_loss
from dataloaders.helpers import *
from mypath import Path

# Setting of parameters
if 'SEQ_NAME' not in os.environ.keys():
    seq_name = 'blackswan'
else:
    seq_name = str(os.environ['SEQ_NAME'])

if 'INST' not in os.environ.keys():
    inst = 1
else:
    inst = int(os.environ['INST'])

if 'COUNT' not in os.environ.keys():
    count = 5
else:
    count = int(os.environ['COUNT'])

if 'SAVE_DIR' not in os.environ.keys():
    save_dir = './experiments'
else:
    save_dir = str(os.environ['SAVE_DIR'])

# Select which GPU, -1 if CPU
if 'GPU' not in os.environ.keys():
    gpu_id = 0
else:
    gpu_id = str(os.environ['GPU'])

db_root_dir = Path.db_root_dir()

# snapshot dir
if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))

vis_net = 0  # Visualize the network?
vis_res = 0  # Visualize the results?
nAveGrad = 5  # Average the gradient every nAveGrad iterations
nEpochs = 2000 * nAveGrad  # Number of epochs for training
snapshots = [iter_ * nAveGrad for iter_ in [1, 10, 100, 1000]] # Store a model every snapshot epochs
parentEpoch = 240

# Parameters in p are used for the name of the model
p = {
    'trainBatch': 1,  # Number of Images in each mini-batch
    }
seed = 1337

parentModelName = 'parent'

# Network definition
# TODO(shelhamer) double-check alignment with our Caffe VGG arch
net = vo.OSVOS(pretrained=0)  # parent network pre-trained on DAVIS fg-bg
net.load_state_dict(torch.load(os.path.join('./models', parentModelName+'_epoch-'+str(parentEpoch-1)+'.pth'),
                               map_location=lambda storage, loc: storage))

if gpu_id >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    net.cuda()

# Visualize the network
if vis_net:
    from util import visualize as viz
    x = torch.randn(1, 3, 480, 854)
    x = Variable(x)
    if gpu_id >= 0:
        x = x.cuda()
    y = net.forward(x)
    g = viz.make_dot(y, net.state_dict())
    g.view()


# Use the following optimizer
lr = 1e-8
wd = 0.0002
optimizer = optim.SGD([
    {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
    {'params': [pr[1] for pr in net.stages.named_parameters() if 'bias' in pr[0]], 'lr': lr * 2},
    {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
    {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': lr*2},
    {'params': [pr[1] for pr in net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0},
    {'params': [pr[1] for pr in net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0},
    {'params': net.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
    {'params': net.fuse.bias, 'lr': 2*lr/100},
    ], lr=lr, momentum=0.9)

# Preparation of the data loaders
# Define augmentation transformations as a composition
composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                          tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                          tr.ToTensor()])
# Training dataset and its iterator
db_train = db.DAVIS2017(train=True, db_root_dir=db_root_dir, transform=composed_transforms, seq_name=seq_name, inst=inst, count=count)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=1)

# Testing dataset and its iterator
db_test = db.DAVIS2017(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name, inst=inst, count=count)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)


num_img_tr = len(trainloader)
num_img_ts = len(testloader)
loss_tr = []
aveGrad = 0

print("Start of Online Training, sequence: " + seq_name)
start_time = timeit.default_timer()
# Main Training and Testing Loop
for epoch in range(0, nEpochs):
    # One training epoch
    running_loss_tr = 0
    np.random.seed(seed + epoch)
    for ii, sample_batched in enumerate(trainloader):

        inputs, gts = sample_batched['image'], sample_batched['gt']

        # Forward-Backward of the mini-batch
        inputs, gts = Variable(inputs), Variable(gts)
        if gpu_id >= 0:
            inputs, gts = inputs.cuda(), gts.cuda()

        outputs = net.forward(inputs)

        # Compute the fuse loss
        loss = class_balanced_cross_entropy_loss(outputs[-1], gts, size_average=False)
        running_loss_tr += loss.data[0]

        # Print stuff
        if nEpochs//20 and epoch % (nEpochs//20) == (nEpochs//20 - 1):
            running_loss_tr /= num_img_tr
            loss_tr.append(running_loss_tr)

            print('[Epoch: %d, numImages: %5d]' % (epoch+1, ii + 1))
            print('Loss: %f' % running_loss_tr)
            print('data/total_loss_epoch {} {}'.format(running_loss_tr, epoch))

        # Backward the averaged gradient
        loss /= nAveGrad
        loss.backward()
        aveGrad += 1

        # Update the weights once in nAveGrad forward passes
        if aveGrad % nAveGrad == 0:
            print('data/total_loss_iter {} {}'.format(loss.data[0], ii + num_img_tr * epoch))
            optimizer.step()
            optimizer.zero_grad()
            aveGrad = 0

    # Save the model
    if epoch + 1 in snapshots:
        torch.save(net.state_dict(), os.path.join("{save_dir}/{seq_name}_{inst}_epoch-{epoch}.pth".format(save_dir=save_dir, seq_name=seq_name, inst=inst, epoch=epoch + 1)))


        print('Testing Network')
        # outputs dir
        save_dir_res = os.path.join(save_dir, "iter{}/{}_{}".format(str((epoch + 1)// nAveGrad), seq_name, inst))
        if not os.path.exists(save_dir_res):
            os.makedirs(save_dir_res)

            # Main Testing Loop
            for ii, sample_batched in enumerate(testloader):

                img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

                # Forward of the mini-batch
                inputs, gts = Variable(img, volatile=True), Variable(gt, volatile=True)
                if gpu_id >= 0:
                    inputs, gts = inputs.cuda(), gts.cuda()

                outputs = net.forward(inputs)

                for jj in range(int(inputs.size()[0])):
                    pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                    pred = 1 / (1 + np.exp(-pred))
                    pred = np.squeeze(pred)

                    # Save the result, attention to the index jj
                    sm.imsave(os.path.join(save_dir_res, os.path.basename(fname[jj]) + '.png'), pred)

                    if vis_res:
                        img_ = np.transpose(img.numpy()[jj, :, :, :], (1, 2, 0))
                        gt_ = np.transpose(gt.numpy()[jj, :, :, :], (1, 2, 0))
                        gt_ = np.squeeze(gt)
                        # Plot the particular example
                        ax_arr[0].cla()
                        ax_arr[1].cla()
                        ax_arr[2].cla()
                        ax_arr[0].set_title('Input Image')
                        ax_arr[1].set_title('Ground Truth')
                        ax_arr[2].set_title('Detection')
                        ax_arr[0].imshow(im_normalize(img_))
                        ax_arr[1].imshow(gt_)
                        ax_arr[2].imshow(im_normalize(pred))
                        plt.pause(0.001)
