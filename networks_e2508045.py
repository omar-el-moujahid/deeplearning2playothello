import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

from tqdm import tqdm
import numpy as np
import os
import time
import matplotlib.pyplot as plt  


def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets)


class MLP(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP, self).__init__()  
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.lin2 = nn.Linear(256,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.lin3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.lin4 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): Board state tensor of shape
            (batch, board_size, board_size) or already flattened.

        Returns:
        - torch.Tensor: Logits of shape (batch, board_size * board_size)
        """
        # Ensure tensor, not numpy, and flatten per sample
        if seq.dim() > 2:
            seq = seq.view(seq.size(0), -1)
        else:
            seq = seq.view(seq.size(0), -1)

        x = self.lin1(seq)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.lin2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.lin3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        outp = self.lin4(x)          # logits, no softmax here
        return outp

    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0
        train_acc_list=[]
        dev_acc_list=[]
        train_loss_list=[]
        dev_loss_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            avg_train_loss = loss_batch / nb_batch
            train_loss_list.append(avg_train_loss)
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+ str(avg_train_loss))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            dev_loss = 0.0
            nb_batch = 0
            for batch, labels, _ in dev:
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                dev_loss += loss.item()
                nb_batch += 1
            avg_dev_loss = dev_loss / nb_batch
            dev_loss_list.append(avg_dev_loss)
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        # Plot training & validation loss and accuracy
        epochs = range(1, len(train_acc_list) + 1)

        plt.figure(figsize=(14, 6))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss_list, label='Train Loss')
        plt.plot(epochs, dev_loss_list, label='Dev Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc_list, label='Train Accuracy')
        plt.plot(epochs, dev_acc_list, label='Dev Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return train_loss_list, dev_loss_list, train_acc_list, dev_acc_list
    
    def evalulate(self,test_loader, device):
        all_predicts=[]
        all_targets=[]
        
        for data, target_array,lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().clone().detach().numpy()
            target=target_array.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep

    

class LSTMs(nn.Module):
    def __init__(self, conf):
        super(LSTMs, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        # LSTM layers
        self.lstm = nn.LSTM(
            self.board_size * self.board_size, 
            self.hidden_dim,
            num_layers=3,
            batch_first=True, 
            dropout=0.3,  
            bidirectional=False
        )
        
        # Fully connected layers
        self.dropout = nn.Dropout(p=0.3)
        self.hidden2output = nn.Linear(self.hidden_dim, self.board_size * self.board_size)

    def forward(self, seq):
        # Handle different input shapes
        if seq.dim() == 4:  # (batch, seq_len, height, width)
            seq = seq.flatten(start_dim=2)  # (batch, seq_len, 64)
        elif seq.dim() == 3:  # (seq_len, height, width)
            seq = seq.flatten(start_dim=1)  # (seq_len, 64)
            seq = seq.unsqueeze(0)  # (1, seq_len, 64) - add batch dimension
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(seq)
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Apply dropout and output layer
        x = self.dropout(last_output)
        outp = self.hidden2output(x)
        
        # Return logits (no activation)
        # CrossEntropyLoss expects raw logits
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0
        train_acc_list=[]
        dev_acc_list=[]
        train_loss_list=[]
        dev_loss_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            avg_train_loss = loss_batch / nb_batch
            train_loss_list.append(avg_train_loss)
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+ str(avg_train_loss))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            dev_loss = 0.0
            nb_batch = 0
            for batch, labels, _ in dev:
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                dev_loss += loss.item()
                nb_batch += 1
            avg_dev_loss = dev_loss / nb_batch
            dev_loss_list.append(avg_dev_loss)
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")
        # Plot training & validation loss and accuracy
        epochs = range(1, len(train_acc_list) + 1)

        plt.figure(figsize=(14, 6))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss_list, label='Train Loss')
        plt.plot(epochs, dev_loss_list, label='Dev Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc_list, label='Train Accuracy')
        plt.plot(epochs, dev_acc_list, label='Dev Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()

        
        return train_loss_list, dev_loss_list, train_acc_list, dev_acc_list
    
    def evalulate(self,test_loader, device):
        all_predicts=[]
        all_targets=[]
        
        for data, target_array,lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().clone().detach().numpy()
            target=target_array.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep

    
class CNN(nn.Module):


    def __init__(self, conf):


        super(CNN, self).__init__()  
       
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_CNN/" 
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
       
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,padding=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,padding=1)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, self.board_size * self.board_size)
       


    def forward(self, seq):
        
        x = F.relu(self.conv1(seq))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        best_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        train_loss_list = []
        dev_loss_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            avg_train_loss = loss_batch / nb_batch
            train_loss_list.append(avg_train_loss)
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+ str(avg_train_loss))
            last_training = time.time() - start_time

            self.eval()
            
            train_clas_rep = self.evalulate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep = self.evalulate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction = time.time() - last_training - start_time
            
            dev_loss = 0.0
            nb_batch = 0
            for batch, labels, _ in dev:
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                dev_loss += loss.item()
                nb_batch += 1
            avg_dev_loss = dev_loss / nb_batch
            dev_loss_list.append(avg_dev_loss)
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculating the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")
        
        # Plot training & validation loss and accuracy
        epochs = range(1, len(train_acc_list) + 1)

        plt.figure(figsize=(14, 6))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss_list, label='Train Loss')
        plt.plot(epochs, dev_loss_list, label='Dev Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc_list, label='Train Accuracy')
        plt.plot(epochs, dev_acc_list, label='Dev Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
        return train_loss_list, dev_loss_list, train_acc_list, dev_acc_list
    
    def evalulate(self, test_loader, device):
        all_predicts = []
        all_targets = []
        
        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep = classification_report(all_targets,
                                        all_predicts,
                                        zero_division=1,
                                        digits=4,
                                        output_dict=True)
        
        return perf_rep