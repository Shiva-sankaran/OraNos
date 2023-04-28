from ast import arg
import imghdr
import torch.nn as nn
import torch
import pdb
import sys
sys.path.append('/home/shape3d/3D-ORS/src')
from encoders.text_encoder import text_encoder_clip
from encoders.obj_encoder import obj_pointnet_encoder
from encoders.sketch_encoder import image_encoder_clip

class ORanoS(nn.Module):
    def __init__(self,args):
        super(ORanoS,self).__init__()
        print("Loading ORanos")
        self.obj_encoder_name = args.obj_encoder
        self.image_encoder_name = args.image_encoder
        self.sketch_encoder_name = args.sketch_encoder

        supported_objEncoder_names = ['pointnet']
        supported_imageEncoders_names = ['clip']
        supported_sketchEncoders_names = ['clip']
        self.args = args
        # if(args.dataparallel):
        #     self.DEVICE = 'cuda'
        # else:
        #     self.DEVICE = 'cuda' + str(args.cuda_id)

        
        # Encoder should be supported
        try:
            assert self.obj_encoder_name in supported_objEncoder_names
        except AssertionError:
            print("Invalid 3D encoder {}, expected one of the following".format(self.obj_encoder_name,supported_objEncoder_names))

        try:
            assert self.image_encoder_name in  supported_imageEncoders_names
        except AssertionError:
            print("Invalid image encoder {}, expected one of the following".format(self.image_encoder_name,supported_imageEncoders_names))

        try:
            assert self.sketch_encoder_name in  supported_sketchEncoders_names
        except AssertionError:
            print("Invalid sketch encoder {}, expected one of the following".format(self.sketch_encoder_name,supported_sketchEncoders_names))

        # Load encoders
        self.image_encoder = self.load_image_encoder(args)#.to(device=self.DEVICE )
        self.obj_encoder = self.load_obj_encoder(args)#.to(device=self.DEVICE )
        self.sketch_encoder = self.load_sketch_encoder(args)#.to(device=self.DEVICE )
        # self.text_encoder = self.load_text_encoder(args)
        # self.obj_encoder.to(self.DEVICE)
        # self.obj_encoder.classifier.to('cuda:1')

        # Load classification heads

        self.classifier_sketch = ClassificationHead(args)
        self.classifier_obj = ClassificationHead(args)
        self.classifier_image = ClassificationHead(args)
        
    def forward(self,obj,image,sketch):

        obj_embeds = self.return_obj_embeds(obj)
        image_embeds = self.return_image_embeds(image)
        sketch_embeds = self.return_sketch_embeds(sketch)

        comb_embeds = sketch_embeds + image_embeds

        obj_class = self.classifier_obj(obj_embeds)
        image_class = self.classifier_image(image_embeds)
        sketch_class = self.classifier_sketch(sketch_embeds)


        return comb_embeds,obj_embeds,sketch_class,image_class,obj_class


    def return_obj_embeds(self,x):
        return self.obj_encoder(x)
    def return_image_embeds(self,x):
        return self.image_encoder(x)
    def return_sketch_embeds(self,x):
        return self.sketch_encoder(x)
    def return_text_embeds(self,x):
        return self.text_encoder(x)

    def load_obj_encoder(self,args):
        if(self.obj_encoder_name == 'pointnet'):
            return obj_pointnet_encoder(args)
    def load_image_encoder(self,args):
        if(self.image_encoder_name == 'clip'):
            return image_encoder_clip(args)
    def load_sketch_encoder(self,args):
        if(self.image_encoder_name == 'clip'):
            return image_encoder_clip(args)
        
    def load_text_encoder(self,args):
        if(self.image_encoder_name == 'clip'):
            return text_encoder_clip(args)



class ClassificationHead(nn.Module):
    def __init__(self, args):
        super(ClassificationHead, self).__init__()

        self.emb_dim = args.emb_dim
        self.hidden_size = args.hidden_size
        self.no_classes = args.no_classes

        self.fc1 = nn.Linear(in_features=self.emb_dim,out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size,out_features=self.no_classes)
        self.relu = nn.ReLU(inplace = True)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.sigmoid(x) # ASL function does sigmoid
        return x

        


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.emb_dim = args.emb_dim
        self.hidden_size = args.hidden_size
        self.no_classes = args.no_classes

        self.fc1 = nn.Linear(in_features=self.emb_dim,out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size,out_features=self.no_classes)
        self.relu = nn.ReLU(inplace = True)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.sigmoid(x) # ASL function does sigmoid
        return x

# class Classifier(nn.Module):
#     def __init__(self, args):
#         super(Classifier, self).__init__()
#         self.sup_cls_size = args.clip_emb_size
#         self.hidden_size = args.classifier_hidden_dim
#         self.no_classes = args.no_classes
        
#         self.fc1 = nn.Linear(in_features=self.sup_cls_size,out_features=self.hidden_size)
#         self.fc2 = nn.Linear(in_features=self.hidden_size,out_features=self.hidden_size)
#         self.fc3 = nn.Linear(in_features=self.hidden_size,out_features=self.no_classes)

#         self.bn1 = nn.BatchNorm1d(self.hidden_size)
#         self.bn2 = nn.BatchNorm1d(self.hidden_size)
#         self.relu = nn.ReLU(inplace = True)
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self,x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x) # ASL function does sigmoid
#         return x

class split_feat(nn.Module):
    def __init__(self, args):
        super(split_feat, self).__init__()
        self.subcls_feat_ext = sub_feat_ext(args=args)
        self.supcls_feat_ext = sup_feat_ext(args=args)
        self.supcls_classifier = Classifier(args=args)
        self.feat_combiner = feat_comb(args=args)
        
    def forward(self,x):
        subcls_feat = self.subcls_feat_ext(x)
        supcls_feat = self.supcls_feat_ext(x)
        feat_combined = self.feat_combiner(supcls_feat,subcls_feat)
        
        # sup_cls = self.supcls_classifier(feat_combined)
        # return subcls_feat,None,feat_combined,None
        return subcls_feat,supcls_feat,feat_combined,None
    def return_sub_cls_feat(self,x):
        return self.subcls_feat_ext(x)
    def return_sup_cls_feat(self,x):
        return self.supcls_feat_ext(x)
    
    
class sup_feat_ext(nn.Module):
    def __init__(self, args):
        super(sup_feat_ext, self).__init__()
        self.clip_emb_size = args.clip_emb_size
        self.hidden_size = args.hidden_size
        self.sup_cls_size = args.sup_cls_size
        
        self.fc1 = nn.Linear(in_features=self.clip_emb_size,out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size,out_features=self.sup_cls_size)
        self.relu = nn.ReLU(inplace = True)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class sub_feat_ext(nn.Module):
    def __init__(self, args):
        super(sub_feat_ext, self).__init__()
        self.clip_emb_size = args.clip_emb_size
        self.hidden_size = args.hidden_size
        self.sub_cls_size = args.sub_cls_size
        
        self.fc1 = nn.Linear(in_features=self.clip_emb_size,out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size,out_features=self.sub_cls_size)

        self.bn1 = nn.BatchNorm1d(self.hidden_size)

        self.relu = nn.ReLU(inplace = True)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class feat_comb(nn.Module):
    def __init__(self, args):
        super(feat_comb, self).__init__()
        self.input_emb_dim = args.sup_cls_size + args.sub_cls_size
        self.clip_emb_size = args.clip_emb_size
        
        self.fc1 = nn.Linear(in_features=self.input_emb_dim,out_features=self.input_emb_dim)
        self.fc2 = nn.Linear(in_features=self.input_emb_dim,out_features=self.input_emb_dim)
        self.fc3 = nn.Linear(in_features=self.input_emb_dim,out_features=self.clip_emb_size)

        self.bn1 = nn.BatchNorm1d(self.input_emb_dim)
        self.bn2 = nn.BatchNorm1d(self.input_emb_dim)
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self,x1,x2):
        x = torch.cat((x1,x2),dim = 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x