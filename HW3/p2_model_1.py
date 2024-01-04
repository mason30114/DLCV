import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import timm
from p2_dataset import P2Dataset
import torchvision.transforms as transforms
from tokenizer import BPETokenizer
import loralib as lora
import loralib as lora
class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        #self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r = 16)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        #self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r = 16)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))
    
class Cross_Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.img_attn = nn.Linear(cfg.n_embd, 2 * cfg.n_embd)                 
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head                            
        self.n_embd = cfg.n_embd

    def forward(self, img_feat, text_feat):
        B, T, C = text_feat.size() # batch, context, embedding    #img_feat = [b, 196+1, 1024] text_feat = [b, 1024, 768]
        B, T1, C = img_feat.size()
        q = self.c_attn(text_feat)                             # q = [b, 1024, 768]
        img_feat = self.img_attn(img_feat)                        # [b, 196+1, 1024] => [b, 196+1, 768], [b, 196+1, 768]
        k, v  = img_feat.split(self.n_embd, dim=2)
        #k = k.view(B, 256+1, self.n_head, C // self.n_head).transpose(1, 2)  #[b, 196+1, 768] => [b, 1, 196+1, 768/1]
        k = k.view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)  #[b, 196+1, 768] => [b, 1, 196+1, 768/1]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #[b, 1024, 1, 768/1] => [b, 1, 1024, 768/1]
        v = v.view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)
        #v = v.view(B, 256+1, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # [b, 1, 1024, 768/1] @ [b, 1, 768/1, 196+1] => [b, 1, 1024, 196+1]
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))  # [b, 1, 1024, 196+1] @ [b, 1, 196+1, 768/1].trans.view.proj => [b, 1024, 768]
    

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)                                  #=====
        self.cross_attn = Cross_Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
             #'c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r = 16)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
             #'c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r = 16))
        ]))

    def forward(self, img_feat, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(img_feat, self.ln_3(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        #self.img_attn = nn.Linear(192, cfg.n_embd)
        self.img_attn = nn.Linear(1408, cfg.n_embd)
        #self.img_attn = nn.Linear(1664, cfg.n_embd)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            #img_attn = nn.Linear(192, cfg.n_embd),
            #img_attn = nn.Linear(1280, cfg.n_embd),
            #img_attn = nn.Linear(1280, 2*cfg.n_embd),
            #img_attn = nn.Linear(1408, cfg.n_embd),
            #img_attn = nn.Linear(1664, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)          ###
        self.transformer.wte.weight = self.lm_head.weight
        #self.lm_head.weight = self.transformer.wte.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)
        #self.lm_head.weight = self.transformer.wte.weight

    def forward(self, img_feat, x):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        img_feat = self.img_attn(img_feat)
        for layer in self.transformer.h:
            x = layer(img_feat, x)
        x = self.lm_head(self.transformer.ln_f(x))
        return x
class Transfomer(nn.Module):

    def __init__(self, encoder, decoder_checkpoint):
        super().__init__()
 
        self.config = Config(checkpoint = decoder_checkpoint)
        self.decoder = Decoder(cfg = self.config)
        self.encoder = timm.create_model(encoder, pretrained=True)                                        #[1, 1000]
                                       #[1, 196, 1024]

    def forward(self, img, cap):
        #img_feat = self.encoder.forward_features(img)[:, 1:257, :]
        img_feat = self.encoder.forward_features(img)
        #print(img_feat.shape)
        cap_out = self.decoder(img_feat, cap)
        return cap_out
    
def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state, strict=False)
    print('model loaded from %s' % checkpoint_path)    

if __name__ == '__main__':
    decoder_checkpoint = './hw3_data/p2_data/decoder_model.bin'
    encoder = 'vit_gigantic_patch14_clip_224'
    #encoder = 'vit_base_patch32_224_in21k'
    img_root = './hw3_data/p2_data/images/train/'
    json_fn = './hw3_data/p2_data/train.json'
    encoder_file = './encoder.json'
    vocab_file = 'vocab.bpe'
    dataset = P2Dataset(img_root, json_fn, transforms.Compose([
                                                               transforms.ToTensor(),
                                                               transforms.Resize((224, 224))
                                                              ]), mode = 'train')
    tokenizer = BPETokenizer(encoder_file = encoder_file, vocab_file = vocab_file)
    model1 = Transfomer(encoder, decoder_checkpoint)
    model2 = Transfomer(encoder, decoder_checkpoint)
    dat = dataset[0]
    img = dat[0].unsqueeze(0)
    cont = dat[1].unsqueeze(0)
    gt = dat[2].unsqueeze(0)
    im = torch.ones(1, 3, 224, 224)
    capr = torch.ones(1, 1).long()
    #model2 = Transfomer(encoder, decoder_checkpoint)
    load_checkpoint('./p2_lora_ckpt/gigan16_best.pth', model1)
    #load_checkpoint('./p2_lora_ckpt/gigan16_best.pth', model2)
    #t1 = model1.encoder.forward_features(im)
    #t2 = model2.encoder.forward_features(im)
    #print(t1)
    #print(t2)
    lora.mark_only_lora_as_trainable(model1)
    for param in model1.decoder.img_attn.parameters():
        param.requires_grad = True  
    for layer in model1.decoder.transformer.h: 
        for param in layer.cross_attn.parameters():
            param.requires_grad = True 
        for param in layer.ln_3.parameters():
            param.requires_grad = True 
    n1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(n1)            
    #print(model1(im, capr))
    with torch.no_grad():
        model1.eval()
        n2 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(n1-n2)
    #print(model2.decoder(t2, capr))
    #print(model1.decoder.img_attn.weight)
    #print(model2.decoder.img_attn.weight)
    #print(model1.parameters() == model2.parameters())

    #print(f"## Model #param={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M")
    #load_checkpoint('./p2_lora_ckpt/gigan16_ep0.pth', model)
    #state_dict = torch.load('./hw3_data/p2_data/decoder_model.bin')
    #print(state_dict.keys())

    



    