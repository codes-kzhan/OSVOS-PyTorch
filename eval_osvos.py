import os
import numpy as np
from PIL import Image

data_root = './data/davis'
count = 5
gpu_id = 7

# generate list of video-instance tuples to evaluate on
videos = open(f"{data_root}/ImageSets/2017/val.txt").read().splitlines()
vid_inst = []
for vid in videos:
    labels = np.sort(os.listdir(os.path.join(data_root, 'Annotations/480p/', vid)))
    labels = np.array(Image.open(os.path.join(data_root, 'Annotations/480p/', vid, labels[0])), dtype=np.uint8)
    instances = [x for x in np.unique(labels) if x != 0 and x != 255]
    vid_inst += [(vid, inst) for inst in instances]


os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.environ['COUNT'] = str(count)
for (vid, inst) in vid_inst:
    print(f"Evaluating video {vid}, instance {inst}, with {count} sparse points on first frame.")
    os.environ['SEQ_NAME'] = str(vid)
    os.environ['INST'] = str(inst)
    exec(open("./train_online.py").read())



