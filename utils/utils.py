import os
import sys
import time
import torch
import random
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from PIL import Image
# import pycocotools.coco as coco   
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPTokenizer
import json
import glob
import pandas as pd

sys.path.append('/home/shape3d/3D-ORS')

# class ShapenetDataset(Dataset):
#     def __init__(self,split,args) -> None: # args.shapenet_data_path = '/home/shape3d/data/ShapeNet/'
#         super(ShapenetDataset,self).__init__()
#         print("Loading shapenet")
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
#         # self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

#         self.obj_paths = []
#         self.clss = []
#         self.sketch_paths = []
#         self.image_paths = []
#         self.split = split

#         self.return_mean_cls_embeds = args.return_cls_mean_embeds

#         if(self.return_mean_cls_embeds):
#             self.mean_embeddings = {}
#             for emb_path in os.listdir('/home/shape3d/3D-ORS/mean_embeds'):
#                 cls_name = emb_path.split('_')[0]
#                 mean_emb = np.load('/home/shape3d/3D-ORS/mean_embeds' + '/' + emb_path)
#                 self.mean_embeddings[cls_name] = mean_emb

#         with open( args.shapenet_data_path + 'metadata.yaml','r') as f:
#             meta_data = json.load(f)

#         self.cat2idx = dict()
#         self.idx2cat = dict()
        

#         for k,item in meta_data.items():
            
#             self.cat2idx[item['name'].split(',')[0]] = len(self.cat2idx)
#             self.idx2cat[len(self.idx2cat)] = item['name'].split(',')[0]

#         for dir in os.listdir(args.shapenet_data_path):
#             if('.yaml' in dir):
#                 continue
#             if('json' in dir):
#                 continue
#             cls = meta_data[dir]['name'].split(',')[0]
#             dir_path = args.shapenet_data_path + dir + '/'
#             with open(dir_path + split +'.lst','r') as f:
#                 objs_folders = f.readlines()
#             for fname in objs_folders:
#                 fname = fname.rstrip()
#                 image_path = random.choice(list(glob.glob(dir_path + fname + '/' + "img_choy2016/*.jpg")))
#                 sketch_path = random.choice(list(glob.glob(dir_path + fname + '/' + "sketch/*.png")))
#                 obj_path = dir_path + fname + '/pointcloud.npz'
#                 self.obj_paths.append(obj_path)
#                 self.sketch_paths.append(sketch_path)
#                 self.image_paths.append(image_path)
#                 self.clss.append(self.cat2idx[cls])

#         self.num_samples = len(self.obj_paths)
#         self.num_classes = args.no_classes
#         print("Loaded shapenet")

#     def __getitem__(self, index):
#         obj_path = self.obj_paths[index]
#         sketch_path = self.sketch_paths[index]
#         image_path = self.image_paths[index]
#         cls_name = image_path.split('/')[-4]
#         cls_int = self.clss[index]
#         cls = np.zeros((self.num_classes))
#         cls[cls_int] = 1

#         obj_array = np.load(obj_path)['points'].transpose(1,0).astype(np.float32)
#         choice = np.random.choice(len(obj_array[0]), 5000, replace=True)
#         # #resample
#         obj_array = obj_array[:, choice]
#         obj_array = obj_array - np.expand_dims(np.mean(obj_array, axis = 0), 0) # center
#         dist = np.max(np.sqrt(np.sum(obj_array ** 2, axis = 1)),0)
#         obj_array = obj_array / dist #scale
#         obj_array = torch.from_numpy(obj_array)

#         sketch = Image.open(sketch_path)
#         sketch = self.processor(images=sketch, padding=True, return_tensors="pt")
#         sketch = sketch['pixel_values'][0]

#         image = Image.open(image_path)
#         image = self.processor(images=image, padding=True, return_tensors="pt")
#         image = image['pixel_values'][0]

#         if(self.return_mean_cls_embeds):
#             cls_emb = torch.Tensor(self.mean_embeddings[cls_name])
#         else:
#             cls_emb = None

#         if(self.split == 'test'):
#             return {'sketch': sketch, 'image': image, 'obj': obj_array, 'cls': cls,'sub_classes':'/'.join(obj_path.split('/')[:-1]),'cls_emb':cls_emb}
#         return {'sketch': sketch, 'image': image, 'obj': obj_array, 'cls': cls,'cls_emb':cls_emb}

#     def __len__(self):
#         return self.num_samples

class TableChairs(Dataset):
    def __init__(self,split,args) -> None: # args.shapenet_data_path = '/home/shape3d/data/ShapeNet/'
        super(TableChairs,self).__init__()
        print("Loading shapenet-TC")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        # self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

        self.obj_paths = []
        self.clss = []
        self.sketch_paths = []
        self.image_paths = []
        self.split = split

        table_chairs_subclasses = os.listdir('/home/shape3d/data/tablechairs/nrrd_256_filter_div_128_solid')
        tc_captions = pd.read_csv('/home/shape3d/data/tablechairs/captions.tablechair.csv')
        all_sub_classes = tc_captions['modelId'].tolist()
        

        with open( args.shapenet_data_path + 'metadata.yaml','r') as f:
            meta_data = json.load(f)

        self.cat2idx = dict()
        self.idx2cat = dict()
        self.return_mean_cls_embeds = False

        for k,item in meta_data.items():
            _cls_name = item['name'].split(',')[0] 
            if( not (_cls_name == 'table' or _cls_name== 'chair')):
                continue
            self.cat2idx[item['name'].split(',')[0]] = len(self.cat2idx)
            self.idx2cat[len(self.idx2cat)] = item['name'].split(',')[0]

        for dir in os.listdir(args.shapenet_data_path):
            if('.yaml' in dir):
                continue
            if('json' in dir):
                continue
            cls = meta_data[dir]['name'].split(',')[0]
            if( not (cls == 'table' or cls== 'chair')):
                continue
            dir_path = args.shapenet_data_path + dir + '/'
            with open(dir_path + split +'.lst','r') as f:
                objs_folders = f.readlines()
            for fname in objs_folders:
                # if(fname not in  all_sub_classes):
                #     continue
                fname = fname.rstrip()
                image_path = random.choice(list(glob.glob(dir_path + fname + '/' + "img_choy2016/*.jpg")))
                sketch_path = random.choice(list(glob.glob(dir_path + fname + '/' + "sketch/*.png")))
                obj_path = dir_path + fname + '/pointcloud.npz'
                self.obj_paths.append(obj_path)
                self.sketch_paths.append(sketch_path)
                self.image_paths.append(image_path)
                self.clss.append(self.cat2idx[cls])

        self.num_samples = len(self.obj_paths)
        self.num_classes = args.no_classes
        print("Loaded shapenet-TC")

    def __getitem__(self, index):
        obj_path = self.obj_paths[index]
        sketch_path = self.sketch_paths[index]
        image_path = self.image_paths[index]
        cls_name = image_path.split('/')[-4]
        cls_int = self.clss[index]
        cls = np.zeros((self.num_classes))
        cls[cls_int] = 1

        obj_array = np.load(obj_path)['points'].transpose(1,0).astype(np.float32)
        choice = np.random.choice(len(obj_array[0]), 5000, replace=True)
        # #resample
        obj_array = obj_array[:, choice]
        obj_array = obj_array - np.expand_dims(np.mean(obj_array, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(obj_array ** 2, axis = 1)),0)
        obj_array = obj_array / dist #scale
        obj_array = torch.from_numpy(obj_array)

        sketch = Image.open(sketch_path)
        sketch = self.processor(images=sketch, padding=True, return_tensors="pt")
        sketch = sketch['pixel_values'][0]

        image = Image.open(image_path)
        image = self.processor(images=image, padding=True, return_tensors="pt")
        image = image['pixel_values'][0]

        if(self.return_mean_cls_embeds):
            cls_emb = torch.Tensor(self.mean_embeddings[cls_name])
        else:
            cls_emb = None

        if(self.split == 'test'):
            return {'sketch': sketch, 'image': image, 'obj': obj_array, 'cls': cls,'sub_classes':'/'.join(obj_path.split('/')[:-1])}
        return {'sketch': sketch, 'image': image, 'obj': obj_array, 'cls': cls}

    def __len__(self):
        return self.num_samples


# class build_dataset(Dataset):
#     def __init__(self, split, args):
#         super(build_dataset, self).__init__()

#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
#         self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
#         self.precompute_txt = args.precompute_txt
        
#         self.data_dir = 'dataset/' + split + '2014/'
#         if split == 'train' or split == 'val':
#             self.annot_path_caption = os.path.join(self.data_dir, 'annotations/captions_' + split + '2014.json')
#             self.annot_path_obj = os.path.join(self.data_dir, 'annotations/instances_' + split + '2014.json')

#         self.sketch_dir = self.data_dir + 'sketches/'
#         self.image_dir = self.data_dir + 'images/'

#         print('==> initializing %s caption data.' % split)
#         self.coco_cap = coco.COCO(self.annot_path_caption)
#         self.images = self.coco_cap.getImgIds()
#         self.num_samples = len(self.images)

#         print('==> initializing %s obj data.' % split)
#         self.coco_obj = coco.COCO(self.annot_path_obj)

#         self.cat2cat = dict()
#         for cat in self.coco_obj.cats.keys():
#             self.cat2cat[cat] = len(self.cat2cat)

#         self.DEVICE = args.cuda_id
#         # print(self.cat2cat)

#         # if args.precompute_txt:
#         #     t1 = time.time()
#         #     self.text_embeds = {}
#         #     for index in range(len(self.images)):
#         #         img_id = self.images[index]
#         #         annotations_cap = self.coco_cap.loadAnns(ids=self.coco_cap.getAnnIds(imgIds=[img_id]))
#         #         texts = [t['caption'] for t in annotations_cap]
#         #         self.text_embeds[img_id] = [self.text_tokenizer(x, padding=True, return_tensors="pt") for x in texts]
#         #     print("Time to precompute text embeddings = {}".format(time.time() - t1))
#         #     self.max_text_len = 65 # max text length = 62. Zero pad at end

#     def __getitem__(self, index):

#         img_id = self.images[index]

#         image = Image.open(os.path.join(self.image_dir, self.coco_cap.loadImgs(ids=[img_id])[0]['file_name']))
#         image = self.processor(images=image, padding=True, return_tensors="pt")
#         image = image['pixel_values'][0]

#         sketch = Image.open(os.path.join(self.sketch_dir, self.coco_cap.loadImgs(ids=[img_id])[0]['file_name'][:-3] + 'png'))
#         sketch = self.processor(images=sketch, padding=True, return_tensors="pt")
#         sketch = sketch['pixel_values'][0]

#         if self.precompute_txt:
#             text_idx = np.random.choice(np.arange(5))
#             text = self.text_embeds[img_id][text_idx]
#             pad_len = self.max_text_len - len(self.text_embeds[img_id][text_idx]['input_ids'][0])
#             pad = torch.zeros((1, pad_len)).to(self.DEVICE)
#             text['input_ids'] = torch.cat((text['input_ids'].to(self.DEVICE), pad), 1)
#             text['attention_mask'] = torch.cat((text['attention_mask'].to(self.DEVICE), pad), 1)
#             text = text.to(self.DEVICE)
#         else:
#             annotations_cap = self.coco_cap.loadAnns(ids=self.coco_cap.getAnnIds(imgIds=[img_id]))
#             texts = [t['caption'] for t in annotations_cap] # choose randomly from 5 captions
#                                                             # might need to change later
#             random.shuffle(texts)
#             text = texts[0]

#         # taken from https://github.com/Alibaba-MIIL/ASL (helper_functions.py)
#         annotations_obj = self.coco_obj.loadAnns(ids=self.coco_obj.getAnnIds(imgIds=[img_id]))
#         output = torch.zeros((3, 80), dtype=torch.long)
#         for obj in annotations_obj:
#             if obj['area'] < 32 * 32:
#                 output[0][self.cat2cat[obj['category_id']]] = 1
#             elif obj['area'] < 96 * 96:
#                 output[1][self.cat2cat[obj['category_id']]] = 1
#             else:
#                 output[2][self.cat2cat[obj['category_id']]] = 1
#         cls = output.to(self.DEVICE)

#         # ----DEBUG: checking inputs----
#         # image = Image.open(os.path.join(self.image_dir, self.coco_cap.loadImgs(ids=[img_id])[0]['file_name']))
#         # image.save('temp_image.png')
#         # sketch = Image.open(os.path.join(self.sketch_dir, self.coco_cap.loadImgs(ids=[img_id])[0]['file_name'][:-3] + 'png'))
#         # sketch.save('temp_sketch.png')
#         # print(texts)
#         # import json 
#         # with open(self.annot_path_obj) as f:
#         #     t1 = json.load(f)
#         # for obj in annotations_obj:
#         #     print(t1['categories'][self.cat2cat[obj['category_id']]])

#         return {'sketch': sketch, 'text': text, 'image': image, 'cls': cls}

#     def __len__(self):

#         return self.num_samples

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

# from https://github.com/mlfoundations/open_clip
def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

# taken from https://github.com/Alibaba-MIIL/ASL
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

# taken from https://github.com/Alibaba-MIIL/ASL
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

loss_query = nn.CrossEntropyLoss()
loss_obj = nn.CrossEntropyLoss()
loss_cls = nn.CrossEntropyLoss() # No multi-class
# loss_L1 = nn.L1Loss()
loss_L1 = nn.KLDivLoss(reduction="batchmean")
# loss_cls = AsymmetricLossOptimized()

def train_one_epoch(oranos, data_loader_train, epoch, optimizer, scheduler, args):
    DEVICE = 'cuda:' + str(args.cuda_id)
    oranos.train()
    avg_epoch_loss, avg_emb_loss, avg_cls_loss = 0, 0, 0
    batch_idx = 0
    for batch in tqdm(data_loader_train):
        # for k in batch:
        #     batch[k] = batch[k].to(device='cuda', non_blocking=True)

        sketch = batch['sketch'].to(device=DEVICE, non_blocking=True)
        image = batch['image'].to(device=DEVICE, non_blocking=True)
        obj = batch['obj'].to(device=DEVICE, non_blocking=True)
        cls = batch['cls'].to(device=DEVICE, non_blocking=True)

        optimizer.zero_grad()
        combined_embs, obj_embs, sketch_class, image_class, obj_class = oranos(obj, image, sketch)
        logits_per_obj = obj_embs @ combined_embs.T
        logits_per_query = logits_per_obj.T

        ground_truth = torch.arange(args.batch_size, dtype = torch.long, device = DEVICE)
        embed_loss = (loss_query(logits_per_query, ground_truth) + loss_obj(logits_per_obj, ground_truth)) / 2
        
        cls = cls.argmax(dim=1)
        cls_loss_sketch = loss_cls(sketch_class, cls)
        cls_loss_image = loss_cls(image_class, cls)
        cls_loss_obj = loss_cls(obj_class, cls)
        cls_loss = (cls_loss_sketch + cls_loss_obj + cls_loss_image) / 3

        total_loss = (args.embed_ratio*embed_loss + args.cls_ratio*cls_loss) / (args.embed_ratio + args.cls_ratio)
        total_loss.backward()
        avg_epoch_loss += total_loss
        avg_emb_loss += embed_loss
        avg_cls_loss += cls_loss

        optimizer.step()
        step = len(data_loader_train) * epoch + batch_idx
        scheduler(step)
        batch_idx +=1

    # print({'epoch': epoch, 'Train total loss': total_loss.item(), 'Train embed loss': embed_loss.item(), 'Train cls loss': cls_loss.item(), 'Train GPT loss': gpt_output.loss})
        

    avg_epoch_loss /= len(data_loader_train)
    avg_emb_loss /= len(data_loader_train)
    avg_cls_loss /= len(data_loader_train)

    print("Epoch {} of {}. Avg loss = {}".format(epoch, args.epochs, avg_epoch_loss))
    return avg_epoch_loss, avg_emb_loss, avg_cls_loss

def evaluate(oranos, data_loader_val, epoch, args):
    DEVICE = 'cuda:' + str(args.cuda_id)
    with torch.no_grad():
        avg_epoch_loss, avg_emb_loss, avg_cls_loss = 0, 0, 0
        for batch in tqdm(data_loader_val):
            # for k in batch:
            #     batch[k] = batch[k].to(device='cuda', non_blocking=True)

            sketch = batch['sketch'].to(device=DEVICE, non_blocking=True)
            image = batch['image'].to(device=DEVICE, non_blocking=True)
            obj = batch['obj'].to(device=DEVICE, non_blocking=True)
            cls = batch['cls'].to(device=DEVICE, non_blocking=True)

            combined_embs, obj_embs, sketch_class, image_class, obj_class = oranos(obj, image, sketch)
            logits_per_obj = obj_embs @ combined_embs.T
            logits_per_query = logits_per_obj.T

            ground_truth = torch.arange(args.batch_size, dtype = torch.long, device = DEVICE)
            embed_loss = (loss_query(logits_per_query, ground_truth) + loss_obj(logits_per_obj, ground_truth)) / 2
            
            cls = cls.argmax(dim=1)
            cls_loss_sketch = loss_cls(sketch_class, cls)
            cls_loss_image = loss_cls(image_class, cls)
            cls_loss_obj = loss_cls(obj_class, cls)
            cls_loss = (cls_loss_sketch + cls_loss_obj + cls_loss_image) / 3

            total_loss = (args.embed_ratio*embed_loss + args.cls_ratio*cls_loss) / (args.embed_ratio + args.cls_ratio)
            avg_epoch_loss += total_loss
            avg_emb_loss += embed_loss
            avg_cls_loss += cls_loss

        avg_epoch_loss /= len(data_loader_val)
        avg_emb_loss /= len(data_loader_val)
        avg_cls_loss /= len(data_loader_val)

        print("Validation @ Epoch {}. Avg loss = {}".format(epoch, avg_epoch_loss))

    return avg_epoch_loss, avg_emb_loss, avg_cls_loss


def train_one_epoch_contra(oranos, data_loader_train, epoch, optimizer, scheduler, args,DEVICE):
    # DEVICE = 'cuda:' + str(args.cuda_id)
    oranos.train()
    avg_epoch_loss, avg_emb_loss, avg_cls_loss = 0, 0, 0
    batch_idx = 0
    criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.1,alpha_weight=0.25)
    for batch in tqdm(data_loader_train):
        # for k in batch:
        #     batch[k] = batch[k].to(device='cuda', non_blocking=True)

        sketch = batch['sketch'].to(device=DEVICE, non_blocking=True)
        image = batch['image'].to(device=DEVICE, non_blocking=True)
        obj = batch['obj'].to(device=DEVICE, non_blocking=True)
        cls = batch['cls'].to(device=DEVICE, non_blocking=True)

        optimizer.zero_grad()
        combined_embs, obj_embs, sketch_class, image_class, obj_class = oranos(obj, image, sketch)
        combined_embs = F.normalize(combined_embs,dim=1)
        obj_embs = F.normalize(obj_embs,dim=1)
        contra_loss = criterion(combined_embs,obj_embs)

        cls = cls.argmax(dim=1)
        cls_loss_sketch = loss_cls(sketch_class, cls)
        cls_loss_image = loss_cls(image_class, cls)
        cls_loss_obj = loss_cls(obj_class, cls)
        cls_loss = (cls_loss_sketch + cls_loss_obj + cls_loss_image) / 3

        total_loss = (args.embed_ratio*contra_loss + args.cls_ratio*cls_loss) / (args.embed_ratio + args.cls_ratio)
        total_loss.backward()
        avg_epoch_loss += total_loss
        avg_emb_loss += contra_loss
        avg_cls_loss += cls_loss

        optimizer.step()
        step = len(data_loader_train) * epoch + batch_idx
        scheduler(step)
        batch_idx +=1

    # print({'epoch': epoch, 'Train total loss': total_loss.item(), 'Train embed loss': embed_loss.item(), 'Train cls loss': cls_loss.item(), 'Train GPT loss': gpt_output.loss})
        

    avg_epoch_loss /= len(data_loader_train)
    avg_emb_loss /= len(data_loader_train)
    avg_cls_loss /= len(data_loader_train)

    print("Epoch {} of {}. Avg loss = {}".format(epoch, args.epochs, avg_epoch_loss))
    return avg_epoch_loss, avg_emb_loss, avg_cls_loss

def evaluate_contra(oranos, data_loader_val, epoch, args,DEVICE):
    # DEVICE = 'cuda:' + str(args.cuda_id)
    criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.1,alpha_weight=0.25)
    with torch.no_grad():
        avg_epoch_loss, avg_emb_loss, avg_cls_loss = 0, 0, 0
        for batch in tqdm(data_loader_val):
            # for k in batch:
            #     batch[k] = batch[k].to(device='cuda', non_blocking=True)

            sketch = batch['sketch'].to(device=DEVICE, non_blocking=True)
            image = batch['image'].to(device=DEVICE, non_blocking=True)
            obj = batch['obj'].to(device=DEVICE, non_blocking=True)
            cls = batch['cls'].to(device=DEVICE, non_blocking=True)

            combined_embs, obj_embs, sketch_class, image_class, obj_class = oranos(obj, image, sketch)
            combined_embs = F.normalize(combined_embs,dim=1)
            obj_embs = F.normalize(obj_embs,dim=1)

            contra_loss = criterion(combined_embs,obj_embs)

          
            
            cls = cls.argmax(dim=1)
            cls_loss_sketch = loss_cls(sketch_class, cls)
            cls_loss_image = loss_cls(image_class, cls)
            cls_loss_obj = loss_cls(obj_class, cls)
            cls_loss = (cls_loss_sketch + cls_loss_obj + cls_loss_image) / 3

            total_loss = (args.embed_ratio*contra_loss + args.cls_ratio*cls_loss) / (args.embed_ratio + args.cls_ratio)
            avg_epoch_loss += total_loss
            avg_emb_loss += contra_loss
            avg_cls_loss += cls_loss

        avg_epoch_loss /= len(data_loader_val)
        avg_emb_loss /= len(data_loader_val)
        avg_cls_loss /= len(data_loader_val)

        print("Validation @ Epoch {}. Avg loss = {}".format(epoch, avg_epoch_loss))

    return avg_epoch_loss, avg_emb_loss, avg_cls_loss

def train_one_epoch_cls(encoders, classifiers, data_loader_train, epoch, optimizers, args):
    for c in classifiers.values():
        c.train()
    for e in encoders.values():
        e.eval()

    # !!! CHECK TRAIN?EVAL STATUS

    avg_obj_loss, avg_sketch_loss, avg_image_loss = 0, 0, 0

    for  batch in tqdm(data_loader_train):
        # for k in batch:
        #     batch[k] = batch[k].to(device='cuda', non_blocking=True)

        sketch = batch['sketch'].to(device='cuda', non_blocking=True)
        image = batch['image'].to(device='cuda', non_blocking=True)
        obj = batch['obj'].to(device='cuda', non_blocking=True)
        cls = batch['cls'].to(device='cuda', non_blocking=True)

        for k,o in optimizers.items():
            o.zero_grad()
        with torch.no_grad():
            obj_embs = encoders['object'](obj)
            sketch_embs = encoders['sketch'](sketch)
            image_embs = encoders['image'](image)

        obj_class = classifiers['object'](obj_embs)
        sketch_class = classifiers['sketch'](sketch_embs)
        image_class = classifiers['image'](image_embs)

        cls = cls.max(dim=1)[0].long()
        cls_loss_obj = loss_cls(obj_class, cls) #/ args.batch_size
        cls_loss_sketch = loss_cls(sketch_class, cls) #/ args.batch_size
        cls_loss_image = loss_cls(image_class, cls) #/ args.batch_size

        cls_loss_obj.backward()
        cls_loss_sketch.backward()
        cls_loss_image.backward()

        avg_obj_loss += cls_loss_obj
        avg_sketch_loss += cls_loss_sketch
        avg_image_loss += cls_loss_image

        # for o in optimizers:
        #     o.step()
        for k,o in optimizers.items():
            o.step()

    avg_obj_loss /= len(data_loader_train)
    avg_sketch_loss /= len(data_loader_train)
    avg_image_loss /= len(data_loader_train)

    print("Epoch {} of {}\n Avg obj loss = {}\n Avg sketch loss = {}\n Avg image loss = {}\n ".format(epoch, args.epochs, avg_obj_loss, avg_sketch_loss, avg_image_loss))

    return avg_obj_loss, avg_sketch_loss, avg_image_loss

def evaluate_cls(encoders, classifiers, data_loader_val, epoch, args):
    with torch.no_grad():
        avg_obj_loss, avg_sketch_loss, avg_image_loss = 0, 0, 0
        for _, batch in tqdm(enumerate(data_loader_val)):
            # for k in batch:
            #     batch[k] = batch[k].to(device='cuda', non_blocking=True)

            sketch = batch['sketch'].to(device='cuda', non_blocking=True)
            image = batch['image'].to(device='cuda', non_blocking=True)
            obj = batch['obj'].to(device='cuda', non_blocking=True)
            cls = batch['cls'].to(device='cuda', non_blocking=True)

            obj_embs = encoders['object'](obj)
            sketch_embs = encoders['sketch'](sketch)
            image_embs = encoders['image'](image)

            obj_class = classifiers['object'](obj_embs)
            sketch_class = classifiers['sketch'](sketch_embs)
            image_class = classifiers['image'](image_embs)
            
            cls = cls.max(dim=1)[0].long()

            cls_loss_obj = loss_cls(obj_class, cls)
            cls_loss_sketch = loss_cls(sketch_class, cls)
            cls_loss_image = loss_cls(image_class, cls)
            # cls_loss = (cls_loss_sketch + cls_loss_text + cls_loss_image) / 3

            avg_obj_loss += cls_loss_obj / args.batch_size
            avg_sketch_loss += cls_loss_sketch / args.batch_size
            avg_image_loss += cls_loss_image / args.batch_size

        avg_obj_loss /= len(data_loader_val)
        avg_sketch_loss /= len(data_loader_val)
        avg_image_loss /= len(data_loader_val)

        print("Validation @ Epoch {}\n Avg image loss = {}\n Avg sketch loss = {}\n Avg image loss = {}\n".format(epoch, avg_obj_loss, avg_sketch_loss, avg_image_loss))

    return avg_obj_loss, avg_sketch_loss, avg_image_loss 



def train_one_epoch_feat_splitter(feat_splitter, image_encoder,data_loader_train, epoch, optimizer,scheduler, args):
    DEVICE = 'cuda:' + str(args.cuda_id)
    feat_splitter.train()
    avg_epoch_loss = 0
    batch_idx = 0
    for batch in tqdm(data_loader_train):
        # for k in batch:
        #     batch[k] = batch[k].to(device='cuda', non_blocking=True)

        image = batch['image'].to(device=DEVICE, non_blocking=True)
        cls = batch['cls'].to(device=DEVICE, non_blocking=True)
        # sup_cls_emb = batch['cls_emb'].to(device=DEVICE, non_blocking=True)

        optimizer.zero_grad()
        image_emb = image_encoder(image)
        _,_,feat_combined,_ = feat_splitter(image_emb)

        cls = cls.argmax(dim=1)
        inp = F.log_softmax(image_emb,dim=1)
        target = F.softmax(feat_combined,dim=1)
        total_loss =loss_L1(inp,target)

        # total_loss = loss_L1(image_emb,feat_combined)

        total_loss.backward()
        avg_epoch_loss += total_loss
        optimizer.step()
        scheduler.step()
    avg_epoch_loss /= len(data_loader_train)

    print("Epoch {} of {}. Avg loss = {}".format(epoch, args.epochs, avg_epoch_loss))
    return avg_epoch_loss

def evaluate_feat_splitter(feat_splitter,image_encoder, data_loader_val, epoch, args):
    DEVICE = 'cuda:' + str(args.cuda_id)
    with torch.no_grad():
        avg_epoch_loss = 0
        batch_idx = 0
        for batch in tqdm(data_loader_val):

            image = batch['image'].to(device=DEVICE, non_blocking=True)
            cls = batch['cls'].to(device=DEVICE, non_blocking=True)
            # sup_cls_emb = batch['cls_emb'].to(device=DEVICE, non_blocking=True)

            image_emb = image_encoder(image)
            _,_,feat_combined,_ = feat_splitter(image_emb)

            cls = cls.argmax(dim=1)
            inp = F.log_softmax(image_emb,dim=1)
            target = F.softmax(feat_combined,dim=1)
            total_loss =loss_L1(inp,target)
            # total_loss = loss_L1(image_emb,feat_combined)
            avg_epoch_loss += total_loss

        avg_epoch_loss /= len(data_loader_val)

    print("Val: Epoch {} of {}. Avg val loss = {}".format(epoch, args.epochs, avg_epoch_loss))
    return avg_epoch_loss



"""
This NTXentLoss implementation is taken from: https://github.com/edreisMD/ConVIRT-pytorch/blob/master/loss/nt_xent.py
"""

import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, alpha_weight):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim = 1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs,
                    norm=True,
                    weights=1.0):
        temperature = self.temperature
        alpha = self.alpha_weight

        """
        Pytorch implementation of the loss  SimCRL function by googleresearch: https://github.com/google-research/simclr
        @article{chen2020simple,
                title={A Simple Framework for Contrastive Learning of Visual Representations},
                author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2002.05709},
                year={2020}
                }
        @article{chen2020big,
                title={Big Self-Supervised Models are Strong Semi-Supervised Learners},
                author={Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2006.10029},
                year={2020}
                }
        """

        LARGE_NUM = 1e9
        """Compute loss for model.
        Args:
        hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.
        tpu_context: context information for tpu.
        weights: a weighting number or vector.
        Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
        """
        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
            
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        labels = labels.to(self.device)
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        
        """
        Different from Image-Image contrastive learning
        In the case of Image-Text contrastive learning we do not compute the similarity function between the Image-Image and Text-Text pairs  
        """
        # logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,0, 1)) / temperature
        # logits_aa = logits_aa - masks * LARGE_NUM
        # logits_bb = torch.matmul(hidden2,  torch.transpose(hidden2_large,0, 1)) / temperature
        # logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large,0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large,0, 1)) / temperature

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)

        return alpha*loss_a + (1-alpha)*loss_b
