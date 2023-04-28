from transformers import CLIPTokenizer, CLIPModel
import torch.nn as nn
import torch
import pdb

class text_encoder_clip(nn.Module):
    def __init__(self, args):
        super(text_encoder_clip, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        # self.precompute_txt = args.precompute_txt

    def forward(self, x, wordlevel=False):
        # if not wordlevel:
        inputs = self.text_tokenizer(x, padding=True, truncation = True,return_tensors="pt")
        # else:
        #     inputs = self.text_tokenizer(x, padding=True, return_tensors="pt")
        #     return torch.stack([self.clip_model.get_text_features(**{'input_ids':x.unsqueeze(dim=-1), 'attention_mask':y.unsqueeze(dim=-1)}) for x, y in zip(inputs['input_ids'], inputs['attention_mask'])], dim=0)
        return self.clip_model.get_text_features(**inputs) 
