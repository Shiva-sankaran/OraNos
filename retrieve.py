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
import pandas as pd
from encoders.text_encoder import text_encoder_clip

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--resume_path', default='checkpoints/main/checkpoint_latest.pth', type=str)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=1, type=int)
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
parser.add_argument('--cuda_id', default=0, type=int)
parser.add_argument('--log_name',default='retrieval')

args = parser.parse_args()
# captions = pd.read_csv('/home/shape3d/data/tablechairs/captions.tablechair.csv')

def check_topK(true_clss,pred_clss,k):
    if(true_clss in pred_clss[:k]):
        return True
    else:
        return False
    

def check_topK_superclass(true_clss,pred_clss,k):
    pred_super_cls = ['']*len(pred_clss)
    true_super_cls = true_clss.split('/')[-2]
    for i in range(len(pred_clss)):
        pred_super_cls[i] = pred_clss[i].split('/')[-2]
    if(true_super_cls in pred_super_cls[:k]):
        return True
    else:
        return False
    

DEVICE = 'cuda:' + str(args.cuda_id)
# CHECKPOINT = "/home/shape3d/3D-ORS/checkpoints/retrieval-contra_120_EPOCHS/checkpoint_latest.pth"#"/home/shape3d/3D-ORS/checkpoints/retrieval-_120_EPOCHS/checkpoint_epoch112.pth"
CHECKPOINT = '/home/shape3d/3D-ORS/checkpoints/TC_contra/checkpoint_epoch240.pth'

oranos = ORanoS(args)
oranos.load_state_dict(torch.load(CHECKPOINT,map_location = 'cpu')['model'])
oranos = oranos.to(DEVICE)


# Dataset = ShapenetDataset
Dataset = TableChairs
dataset_test = Dataset('test', args)
sampler_test = torch.utils.data.RandomSampler(dataset_test)
data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, sampler=sampler_test,drop_last=True,pin_memory=True,num_workers=2)

test_embeds = np.load('/home/shape3d/3D-ORS/stored_embeds/TC_test_embedding_contrastive_epoch240.npy')
test_sub_classes = np.load('/home/shape3d/3D-ORS/stored_embeds/TC_test_sub_classes_contrastive_epoch240.npy')

test_embeds = torch.tensor(test_embeds).to(DEVICE)

match_comb = {1: 0, 5: 0, 10: 0}
match_sketch = {1: 0, 5: 0, 10: 0}
match_image = {1: 0, 5: 0, 10: 0}
text = True
if(text):
    captions = pd.read_csv('/home/shape3d/data/tablechairs/captions.tablechair.csv')
    captions.dropna(inplace = True) 
    text_encoder = text_encoder_clip(args)
TOTAL_NOBJS = 0
with torch.no_grad():
    for batch in tqdm(data_loader_test):
        sub_cls = data_loader_test.dataset.idx2cat[batch['cls'][0].argmax().item()]
        if(sub_cls != 'chair' and sub_cls != 'table'):
            continue

        TOTAL_NOBJS += 1  # Make sure batch size is 1, the code only works for batch size of 1 for now
        sketch = batch['sketch'].to(device=DEVICE, non_blocking=True)
        image = batch['image'].to(device=DEVICE, non_blocking=True)
        sub_classes = batch['sub_classes']
        sub_cls = sub_classes[0]


        sketch_embeds = oranos.return_sketch_embeds(sketch)
        if(text):
            modelID = sub_cls.split('/')[-1]
            if(modelID not in captions['modelId'].values):
                TOTAL_NOBJS-=1
                continue
            caption = random.choice(captions[captions['modelId'] == modelID]['description'].values)
            text_embeds = text_encoder(caption).to('cuda:0')
            comb_embeds = sketch_embeds + text_embeds
            image_embeds = text_embeds
        else:
            image_embeds = oranos.return_image_embeds(image)
            comb_embeds = image_embeds + sketch_embeds

        logits_per_obj_combined = test_embeds @ comb_embeds.T
        logits_per_obj_sketch = test_embeds @ sketch_embeds.T
        logits_per_obj_image = test_embeds @ image_embeds.T

        idxs_comb = torch.topk(logits_per_obj_combined.squeeze(),10).indices # Are these image indices?? or local indices??
        idxs_sketch = torch.topk(logits_per_obj_sketch.squeeze(),10).indices
        idxs_image = torch.topk(logits_per_obj_image.squeeze(),10).indices

        idxs_comb = list(idxs_comb.detach().cpu().numpy())
        idxs_sketch = list(idxs_sketch.detach().cpu().numpy())
        idxs_image = list(idxs_image.detach().cpu().numpy())

        ret_comb = test_sub_classes[idxs_comb]
        ret_sketch = test_sub_classes[idxs_sketch]
        ret_image = test_sub_classes[idxs_image]

        
        for k in [1, 5, 10]:
            if(check_topK(true_clss = sub_cls,pred_clss = ret_comb , k = k)):
                match_comb[k]+=1
        for k in [1, 5, 10]:
            if(check_topK(true_clss = sub_cls,pred_clss = ret_sketch , k = k)):
                match_sketch[k]+=1
        for k in [1, 5, 10]:
            if(check_topK(true_clss = sub_cls,pred_clss = ret_image , k = k)):
                match_image[k]+=1

        # if sub class in ret_{} then TP

print(TOTAL_NOBJS)

print("Metrics for combined embeddings")
for k, v in match_comb.items():
    print("Top {} Accuracy: {}".format(k, v/TOTAL_NOBJS * 100))

print()

print("Metrics for sketch embeddings")
for k, v in match_sketch.items():
    print("Top {} Accuracy: {}".format(k, v/TOTAL_NOBJS * 100))

print()

print("Metrics for image embeddings")
for k, v in match_image.items():
    print("Top {} Accuracy: {}".format(k, v/TOTAL_NOBJS * 100))


