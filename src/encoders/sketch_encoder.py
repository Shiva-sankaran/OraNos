from transformers import CLIPTokenizer, CLIPModel
import torch.nn as nn

class image_encoder_clip(nn.Module):
    def __init__(self,args):
        super(image_encoder_clip, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        # self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    def forward(self, x):
        return self.clip_model.get_image_features(x) 
       