

import torch
from torchvision import models
from preactresnet import preactresnet as preactresnet



_mean = (0.485, 0.456, 0.406)
_std = (0.229, 0.224, 0.225)

mu = torch.tensor(_mean).view(3,1,1).cuda()
std = torch.tensor(_std).view(3,1,1).cuda()


class Model(torch.nn.Module):
    def __init__(self, model:str='resnet18'):
        super().__init__()
        self.norm = lambda x: ( x - mu ) / std
        self.model = None
        if model == 'resnet18':
            self.model =models.resnet18(pretrained=False)
            checkpoint = torch.load("./resnet18_linf_eps8.0.ckpt")
            state_dict_path = 'model'
            sd = checkpoint[state_dict_path]
            sd = {k[len('module.'):]:v for k,v in sd.items()}
            sd_t = {k[len('attacker.model.'):]:v for k,v in sd.items() if k.split('.')[0]=='attacker' and k.split('.')[1]!='normalize'}
            self.model.load_state_dict(sd_t)
            self.model.fc = torch.nn.Identity()
            self.model.fc__ = torch.nn.Linear(512, 7)



        if model == 'resnet50':
            self.model =models.resnet50(pretrained=False)
            checkpoint = torch.load("./resnet50_linf_eps8.0.ckpt")
            state_dict_path = 'model'
            sd = checkpoint[state_dict_path]
            sd = {k[len('module.'):]:v for k,v in sd.items()}
            sd_t = {k[len('attacker.model.'):]:v for k,v in sd.items() if k.split('.')[0]=='attacker' and k.split('.')[1]!='normalize'}
            self.model.load_state_dict(sd_t)
            self.model.fc = torch.nn.Identity()
            self.model.fc__ = torch.nn.Linear(2048, 7)
            
        if model == 'wide_resnet50_2':
            self.model =models.wide_resnet50_2(pretrained=False)
            checkpoint = torch.load("./wide_resnet50_2_linf_eps8.0.ckpt")
            state_dict_path = 'model'
            sd = checkpoint[state_dict_path]
            sd = {k[len('module.'):]:v for k,v in sd.items()}
            sd_t = {k[len('model.'):]:v for k,v in sd.items() if k.split('.')[0]=='model'} 
            self.model.load_state_dict(sd_t)
            self.model.fc = torch.nn.Identity()
            self.model.fc__ = torch.nn.Linear(2048, 7)

        if model == 'preactresnet_18':
            self.model = preactresnet.PreActResNet18()
            self.model.load_state_dict(sd_t)
            self.model.fc = torch.nn.Identity()
            self.model.fc__ = torch.nn.Linear(2048, 7)



  
    def forward(self, x):
        x = self.norm(x)
        z1 = self.model(x)
        z2 = self.model.fc__(z1)
 
        return z2

