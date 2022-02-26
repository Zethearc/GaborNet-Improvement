import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import glob
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.optim import RMSprop
from GaborNet_DogsvsCats import GaborNN
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from dataset import dataset
#Dataset
os.makedirs('C:/Users/User/Desktop/GaborNet Final/dataset', exist_ok=True)
base_dir = 'C:/Users/User/Desktop/GaborNet Final/Files/'
train_dir = 'C:/Users/User/Desktop/GaborNet Final/dataset/train'
test_dir = 'C:/Users/User/Desktop/GaborNet Final/dataset/test1'

# with zipfile.ZipFile('C:/Users/User/Desktop/GaborNet Final/Files/train.zip') as train_zip:
#     train_zip.extractall('C:/Users/User/Desktop/GaborNet Final/dataset')  
# with zipfile.ZipFile('C:/Users/User/Desktop/GaborNet Final/Files/test1.zip') as test_zip:
#     test_zip.extractall('C:/Users/User/Desktop/GaborNet Final/dataset')
train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
train_list, val_list = train_test_split(train_list, test_size=0.3)
train_transforms =  transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transforms =   transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
val_transforms =    transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = dataset(train_list, transform=train_transforms)
test_data = dataset(test_list, transform=test_transforms)
val_data = dataset(val_list, transform=val_transforms)
train_loader64 =  torch.utils.data.DataLoader(dataset = train_data, batch_size=64, shuffle=True)
test_loader64 =   torch.utils.data.DataLoader(dataset = test_data, batch_size=64, shuffle=True)
val_loader64 =    torch.utils.data.DataLoader(dataset = val_data, batch_size=64, shuffle=True)
train_loader32 =  torch.utils.data.DataLoader(dataset = train_data, batch_size=32, shuffle=True)
test_loader32 =   torch.utils.data.DataLoader(dataset = test_data, batch_size=32, shuffle=True)
val_loader32 =    torch.utils.data.DataLoader(dataset = val_data, batch_size=32, shuffle=True)
train_loader16 =  torch.utils.data.DataLoader(dataset = train_data, batch_size=16, shuffle=True)
test_loader16 =   torch.utils.data.DataLoader(dataset = test_data, batch_size=16, shuffle=True)
val_loader16 =    torch.utils.data.DataLoader(dataset = val_data, batch_size=16, shuffle=True)
train_loader8 =  torch.utils.data.DataLoader(dataset = train_data, batch_size=8, shuffle=True)
test_loader8 =   torch.utils.data.DataLoader(dataset = test_data, batch_size=8, shuffle=True)
val_loader8 =    torch.utils.data.DataLoader(dataset = val_data, batch_size=8, shuffle=True)
#Device
device = 'cpu'
print('Dogs vs Cats Dataset')
print('Train Dataset Size =',len(train_data))
print('Validation Dataset Size =',len(val_data))
print('Test Dataset Size =',len(test_data))
'''Experiment 4 Batch = 32, Loss = NLL, Optimizer = Adam, Learning_rate = 0.001'''
#Hyperparameters
model = None
model = GaborNN().to(device)
model.train()
epochs = 10
optimizer = Adam(params = model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()
#Model
final_csv = []
list_headers = ['Epoch','Batch Size','Loss','Optimizer','Train Accuracy','Train Loss', 'Validation Accuracy', 'Validation Accuracy']
final_csv.append(list_headers)
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in train_loader8:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc/len(train_loader8)
        epoch_loss += loss/len(train_loader8)
    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
    with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss =0
        for data, label in val_loader8:
            data = data.to(device)
            label = label.to(device)
            val_output = model(data)
            val_loss = criterion(val_output,label)
            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc/ len(val_loader8)
            epoch_val_loss += val_loss/ len(val_loader8)
        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))
        epoch_results = [epoch+1, 32, 'NLL','Adam',str(epoch_accuracy),str(epoch_loss),str( epoch_val_accuracy),str( epoch_val_loss)]
        final_csv.append(epoch_results)
np.savetxt("8_CEL_ADAM.csv", final_csv, delimiter =",",fmt ='% s')