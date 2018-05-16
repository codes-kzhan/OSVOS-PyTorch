import sys, os
import numpy as np
from PIL import Image
from setproctitle import setproctitle

from train_online import train_osvos_online

data_root = './data/davis'
count = int(sys.argv[1])
gpu_id = int(sys.argv[2])


# generate list of video-instance tuples to evaluate on
videos = open(f"{data_root}/ImageSets/2017/val.txt").read().splitlines()
vid_inst = []
for vid in videos:
    labels = np.sort(os.listdir(os.path.join(data_root, 'Annotations/480p/', vid)))
    labels = np.array(Image.open(os.path.join(data_root, 'Annotations/480p/', vid, labels[0])), dtype=np.uint8)
    instances = [x for x in np.unique(labels) if x != 0 and x != 255]
    vid_inst += [(vid, inst) for inst in instances]


count_ = 'dense' if count == -1 else '{}sparse'.format(str(count))
save_dir = './experiments/{}-short'.format(count_)
setproctitle('OSVOS eval {}'.format(count_))

for (vid, inst) in vid_inst:
    print("Evaluating video {}, instance {}, with {} points on first frame.".format(vid, inst, count_))
    train_osvos_online(vid, inst, count, save_dir, gpu_id)



