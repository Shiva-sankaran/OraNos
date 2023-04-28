import torch
import argparse
import tqdm as tq
import numpy as np
import torch.nn as nn
from torch import optim
from utils.utils import *
from src.oranos import ORanoS
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--resume_path', default='checkpoints/main/checkpoint_latest.pth', type=str)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--encoder', default='clip_16', type=str)
parser.add_argument('--precompute_txt', default=False, type=bool)

# Training setting
parser.add_argument('--freeze_encoder',default=True,type=bool)
parser.add_argument('--freeze_nlp_decoder',default=True,type=bool)
parser.add_argument('--freeze_class_MLP',default=False,type=bool)
parser.add_argument('--MLP_weights',default=None,type=str)
parser.add_argument('--embed_ratio',default=100,type=int, help="weight ratio for contrastive learning task")
parser.add_argument('--cls_ratio',default=10,type=int, help="weight ratio for classification task")
parser.add_argument('--gpt_ratio',default=1,type=int, help="weight ratio for (captioning)")

# CLIP
parser.add_argument('--emb_dim', default=512, type=int,help="output dimension of text and image features")
parser.add_argument("--lr", type=float, default=5.0e-4, help="Learning rate.")
parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")
parser.add_argument("--eps", type=float, default=1.0e-6, help="Adam epsilon.")
parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
parser.add_argument("--warmup", type=int, default=10000, help="Number of steps to warmup for.")

# classfication MLP, (RANDOM VALUES IN DEFAULT TO GET MODEL RUNNING FOR NOW)
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--no_classes', default=2, type=int)
parser.add_argument('--cls_weights_path', default='/home/shape3d/3D-ORS/checkpoints/testcheckpoint_epoch49.pth', type=str)
# Oranos model args

parser.add_argument('--obj_encoder',default="pointnet",type=str)
parser.add_argument('--image_encoder',default="clip",type=str)
parser.add_argument('--sketch_encoder',default="clip",type=str)
parser.add_argument('--pointnet_model', default='/home/shape3d/pointnet/pointnet.pytorch/utils/cls_512/cls_model_49.pth', type=str)

# Shapenet args
parser.add_argument('--shapenet_data_path', default='/home/shape3d/data/ShapeNet/', type=str)

# GPU
parser.add_argument('--cuda_id', default=1, type=int)
parser.add_argument('--log_name',default='retrieval')

args = parser.parse_args()


DEVICE = 'cuda:' + str(args.cuda_id)
# CHECKPOINT = "/home/shape3d/3D-ORS/checkpoints/retrieval-contra_120_EPOCHS/checkpoint_latest.pth"#"/home/shape3d/3D-ORS/checkpoints/retrieval-_120_EPOCHS/checkpoint_epoch112.pth"
CHECKPOINT = '/home/shape3d/3D-ORS/checkpoints/TC_contra/checkpoint_epoch240.pth'

oranos = ORanoS(args)
oranos.load_state_dict(torch.load(CHECKPOINT,map_location = 'cpu')['model'])
oranos.to(DEVICE)


# Dataset = ShapenetDataset
Dataset = TableChairs
dataset_test = Dataset('test', args)
sampler_test = torch.utils.data.RandomSampler(dataset_test)
data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, sampler=sampler_test,drop_last=True,pin_memory=True,num_workers=2)

test_embeds = []
test_sub_classes = []
with torch.no_grad():
    for batch in tqdm(data_loader_test):

        obj = batch['obj'].to(device=DEVICE, non_blocking=True)
        sub_classes = batch['sub_classes']
        obj_embeds = oranos.return_obj_embeds(obj)
        test_embeds.extend(obj_embeds.detach().cpu().numpy())
        test_sub_classes.extend(sub_classes)

test_embeds = np.array(test_embeds)
test_sub_classes = np.array(test_sub_classes)

np.save('/home/shape3d/3D-ORS/stored_embeds/TC_test_embedding_contrastive_epoch240.npy',test_embeds)
np.save('/home/shape3d/3D-ORS/stored_embeds/TC_test_sub_classes_contrastive_epoch240.npy',test_sub_classes)


dummy = 1
