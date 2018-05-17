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

def eval_osvos_online(seq_name, inst, count, save_dir, gpu_id, weights, epoch):
    db_root_dir = Path.db_root_dir()

    # snapshot dir
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    vis_net = 0  # Visualize the network?
    vis_res = 0  # Visualize the results?
    nAveGrad = 5  # Average the gradient every nAveGrad iterations
    nEpochs = 101 * nAveGrad  # Number of epochs for training, default is 2000
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
    net.load_state_dict(torch.load(weights))

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


    # Preparation of the data loaders
    # Define augmentation transformations as a composition
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                              tr.ToTensor()])

    # Testing dataset and its iterator
    db_test = db.DAVIS2017(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name, inst=inst, count=count)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)


    num_img_ts = len(testloader)

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
                np.save(os.path.join(save_dir_res, os.path.basename(fname[jj]) + '.npy'), pred)

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
