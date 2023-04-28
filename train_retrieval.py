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
parser.add_argument('--resume_path', default='/home/shape3d/3D-ORS/checkpoints/retrieval-1/checkpoint_epoch49.pth', type=str)
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--batch_size', default=12, type=int)
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
parser.add_argument('--no_classes', default=13, type=int)
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
parser.add_argument('--log_name',default='retrieval-_120_EPOCHS_fixed')

args = parser.parse_args()
DATAPARALLEL = False
LOG_NAME = args.log_name
ckpt_dir = '/home/shape3d/3D-ORS/checkpoints/' + LOG_NAME + '/'
tensorboard_dir = '/home/shape3d/3D-ORS/tensorboard/' + LOG_NAME
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
    print("--- Created checkpoint directory for the run")
else:
    print("--- Checkpoint directory already exsists")
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
    print("--- Created tensorboard directory for the run")
else:
    print("--- tensorboard directory already exsists")
DEVICE = 'cuda:' + str(args.cuda_id)

oranos = ORanoS(args)
if(args.resume):
    print("Resuming training")
    saved_model = torch.load(args.resume_path)
    ORANOS_STATE_DICT = saved_model['model']
    OPTIM_STATE_DICT = saved_model['optimizer']
    EPOCH = saved_model['epoch']
    print("Resuming from ", EPOCH)
if(args.resume):
    oranos.load_state_dict(ORANOS_STATE_DICT)
    print("Loaded statedict of oranos")
else:

    """
    best sketch: Epoch 6
    best obj: Epoch 28
    best img: Epoch 3

    """
    best_sketch_epoch = 6
    best_obj_epoch = 28
    best_img_epoch = 3
    PREFIX = "/home/shape3d/3D-ORS/checkpoints/run-1/checkpoint_epoch{}.pth"
    cls_weights_path_sketch = PREFIX.format(best_sketch_epoch)
    cls_weights_path_obj = PREFIX.format(best_obj_epoch)
    cls_weights_path_img = PREFIX.format(best_img_epoch)

    print("Loading sketch classifier weights from: ",cls_weights_path_sketch)
    print("Loading object classifier weights from: ",cls_weights_path_obj)
    print("Loading image classifier weights from: ",cls_weights_path_img)

    cls_checkpoint_sketch = torch.load(cls_weights_path_sketch,map_location='cpu')
    oranos.classifier_sketch.load_state_dict(cls_checkpoint_sketch['model_sketch'])

    cls_checkpoint_obj = torch.load(cls_weights_path_obj)
    oranos.classifier_obj.load_state_dict(cls_checkpoint_obj['model_obj'])

    cls_checkpoint_img = torch.load(cls_weights_path_img)
    oranos.classifier_image.load_state_dict(cls_checkpoint_img['model_image'])

# cls_checkpoint = torch.load(args.cls_weights_path)
# oranos.classifier_sketch.load_state_dict(cls_checkpoint['model_sketch'])
# oranos.classifier_obj.load_state_dict(cls_checkpoint['model_obj'])
# oranos.classifier_image.load_state_dict(cls_checkpoint['model_image'])
if(DATAPARALLEL):
    oranos= nn.DataParallel(oranos)
    DEVICE = torch.device('cuda')
    oranos = oranos.to(DEVICE)
else:
    oranos = oranos.to(DEVICE)

Dataset = ShapenetDataset
dataset_train = Dataset('train', args)
dataset_val = Dataset('val', args)

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.RandomSampler(dataset_val)

data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train,drop_last = True,pin_memory=True,num_workers=2)
data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, sampler=sampler_val,drop_last=True,pin_memory=True,num_workers=2)

# from https://github.com/mlfoundations/open_clip (train.py)
exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
include = lambda n, p: not exclude(n, p)
named_parameters = list(oranos.named_parameters())
gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

for param in oranos.obj_encoder.classifier.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(
    [
        {"params": gain_or_bias_params, "weight_decay": 0.},
        {"params": rest_params, "weight_decay": args.wd},
    ],
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    eps=args.eps,
)

if(args.resume):
    optimizer.load_state_dict(OPTIM_STATE_DICT)
    print("Loaded optimizer")

total_steps = len(data_loader_train) * args.epochs
scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)



writer = SummaryWriter(tensorboard_dir)
print("--- Intiliased tensorboard for logging")

start_epoch = 0
if(args.resume):
    start_epoch = EPOCH + 1
    args.epochs += start_epoch
    print("Start epoch {} and end epoch {}".format(start_epoch,args.epochs))


for epoch in range(start_epoch, args.epochs):

    oranos.train()
    total_loss, emb_loss, cls_loss  = train_one_epoch(oranos, data_loader_train, epoch, optimizer, scheduler, args)

    oranos.eval()
    val_total_loss, val_emb_loss, val_cls_loss = evaluate(oranos, data_loader_val, epoch, args)

    writer.add_scalars('pretrain', {'epoch': epoch, 'Train total loss': total_loss, 'Train embed loss': emb_loss, 
            'Train cls loss': cls_loss, 'Val total loss': val_total_loss, 
            'Val embed loss': val_emb_loss, 'Val cls loss': val_cls_loss}, epoch)
     
    print({'epoch': epoch, 'Train total loss': total_loss, 'Train embed loss': emb_loss, 'Train cls loss': cls_loss,'Val total loss': val_total_loss, 
            'Val embed loss': val_emb_loss, 'Val cls loss': val_cls_loss})

    torch.save({
                'epoch': epoch,
                'model': oranos.state_dict(),
                'optimizer': optimizer.state_dict()
                }, ckpt_dir + 'checkpoint_epoch' + str(epoch) + '.pth')

