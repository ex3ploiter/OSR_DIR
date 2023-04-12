import torch

import numpy as np
from torchvision.datasets import ImageFolder




from torchvision import transforms
from torchvision.datasets import FashionMNIST

import torchvision.transforms as transforms

import numpy as np

from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torchattacks import FGSM ,AutoAttack
from torchvision.utils import save_image
from torchvision.utils import make_grid
import os
import numpy as np
import torch
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import os
import os
import numpy as np
import os
from torch.utils.data import DataLoader
import os
import numpy as np
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision 




transform_32 = transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor()
        ])
transform_224 = transforms.Compose([
            transforms.Resize([224,224]),
        ])
transform_32_tensor = transforms.Compose([
            transforms.Resize([224,224]),
        ])


def show_samples(x):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.show()
def image_grid(x):
  size =224# config.data.image_size train_model
  channels =3# config.data.num_channels
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img
 

 

class MyDataset_Binary(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x, labels,transform=None):
        'Initialization'
        super(MyDataset_Binary, self).__init__()
        self.labels = labels
        self.x = x
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        'Generates one sample of data'
        if self.transform is None:
          x =  self.x[index]
          y = self.labels[index]
        else:
          x = self.transform(self.x[index])
          y = self.labels[index]
       
        return x, y
       


def getTrainSet(labelsToKeep):
    Fashion_train =FashionMNIST('/mnt/sda3/home/ma_hadi/ExposureExperiment_robust/.data/Datasets/FashionMNIST', train=True, download=False, transform=transform_32)
    #######
    train_labels=[]
    images_train=[]
    for x,y in Fashion_train:
        if y in labelsToKeep:
            train_labels.append(labelsToKeep.index(y))
            images_train.append(x) 
            
    exposureDataset = torchvision.datasets.ImageFolder("./one_class_train", transform=transform_32)
    exposureSamples, _ = next(iter(DataLoader(exposureDataset, batch_size=5000, shuffle=True))) 
    for x in exposureSamples:
        train_labels.append(6)
        images_train.append(x)
        
    images_train_t=torch.stack(images_train)                   
    train_data_set_ =MyDataset_Binary(images_train_t,train_labels ,transform_32_tensor)
    
    return train_data_set_
    
                
def getTestSet(labelsToKeep):
    Fashion_test =FashionMNIST('/mnt/sda3/home/ma_hadi/ExposureExperiment_robust/.data/Datasets/FashionMNIST', train=False, download=False, transform=transform_32)

    test_labels=[]
    images_test=[]
    for x,y in Fashion_test:
      if y in labelsToKeep:
          test_labels.append(labelsToKeep.index(y))
          images_test.append(x)
      else:
          test_labels.append(6)
          images_test.append(x)
    images_test=torch.stack(images_test)

    test_data_set_ =MyDataset_Binary(images_test,test_labels ,transform_32_tensor)
    
    return test_data_set_


def getLoaders(labelsToKeep):
    train_loader = DataLoader(getTrainSet(labelsToKeep), batch_size=8, shuffle=True)
    test_loader = DataLoader(getTestSet(labelsToKeep), batch_size=8, shuffle=False)
    
    return train_loader,test_loader
    
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
 
def auc_softmax_adversarial(model, test_loader, test_attack, num_classes):
    
  soft = torch.nn.Softmax(dim=1)
  anomaly_scores = []
  test_labels = []
  model.eval()
  print('AUC Adversarial Softmax Started ...')
  with tqdm(test_loader, unit="batch") as tepoch:
    torch.cuda.empty_cache()
    for i, (data, target) in enumerate(tepoch):
      
      data, target = data.to(device), target.to(device)
      labels = target.to(device)
      adv_data =  test_attack(data, target)
      output = model(adv_data)
      probs = soft(output).squeeze()
      anomaly_scores += probs[:, num_classes].detach().cpu().numpy().tolist()
      test_labels += target.detach().cpu().numpy().tolist()

  test_labels = [t == num_classes for t in test_labels]
  #print(test_labels)
  auc = roc_auc_score(test_labels, anomaly_scores)
  print(f'AUC Adversairal - Softmax - score is: {auc * 100}')
  return auc

def auc_softmax(model, test_loader, num_classes):
  model.eval()
  soft = torch.nn.Softmax(dim=1)
  anomaly_scores = []
  test_labels = []
  print('AUC Softmax Started ...')
  with torch.no_grad():
    with tqdm(test_loader, unit="batch") as tepoch:
      torch.cuda.empty_cache()
      for i, (data, target) in enumerate(tepoch):
        data, target = data.to(device), target.to(device)
        output = model(data)
        probs = soft(output).squeeze()
        anomaly_scores += probs[:, num_classes].detach().cpu().numpy().tolist()
        test_labels += target.detach().cpu().numpy().tolist()
#epoch
  #print(test_labels)
  #print(num_classes)
  test_labels = [t == num_classes for t in test_labels]
  #print(test_labels)
  auc = roc_auc_score(test_labels, anomaly_scores)
  print(f'AUC - Softmax - score on  is: {auc * 100}')
  
  return auc