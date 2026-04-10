import torch
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader,TensorDataset

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X,y = load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

X_train_scaled_tensor = torch.Tensor(X_train_scaled)
X_test_scaled_tensor = torch.Tensor(X_test_scaled)

y_train_tensor = torch.Tensor(y_train).unsqueeze(1)
y_test_tensor = torch.Tensor(y_test).unsqueeze(1)

train_dataset = TensorDataset(X_train_scaled_tensor,y_train_tensor)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)


class BCNet(nn.Module):

    def __init__(self):
        super(BCNet,self).__init__()

        self.fc1 = nn.Linear(30,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))

        return x
    

model = BCNet()
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(),lr=0.001)
        
epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0.0


    for x_batch,y_batch in train_loader:
        optimiser.zero_grad()

        preds = model(x_batch)
        loss = criterion(preds,y_batch)

        loss.backward()
        optimiser.step()

        running_loss += loss.item()

        print(f"epoch {epoch+1}: Loss was {(running_loss) / len(train_loader)}")

