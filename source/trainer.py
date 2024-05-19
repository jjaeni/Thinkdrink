import torch
import torch.nn as nn
from tqdm import tqdm

class ResNet50V2Trainer:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.epoch = 30
        if torch.cuda.is_available():
            self.DEVICE = torch.device('cuda')
        else:
            self.DEVICE = torch.device('cpu')
        self.best_loss = float('inf')
        self.early_stop_counter=0

    def train(self, model, train_loader, optimizer):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training Progress')
        for image, label in progress_bar:
            image = image.to(self.DEVICE)
            label = label.to(self.DEVICE)
            optimizer.zero_grad()
            output = model(image)
            loss = self.criterion(output, label)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
    
    def evaluate(self, model, val_loader):
        model.eval()
        val_loss=0
        correct=0
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(self.DEVICE)
                label = label.to(self.DEVICE)
                output = model(image)
                val_loss += self.criterion(output, label).item()
                prediction = output.max(1, keepdim=True)[1]
                correct += prediction.eq(label.view_as(prediction)).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = 100. * correct / len(val_loader.dataset)
        return val_loss, val_acc
    
    def train_model(self, model, early_stop_epochs):
        for Epoch in range(1, self.epoch+1):
            train(model, train_loader, optimizer)
            test_loss, test_acc = eval(model, test_loader)
            print('\n[EPOCH: {}], \tTest Accuracy: {:.2f}%, \tTest Loss: {:.4f}'.format(Epoch, test_acc, test_loss))

            if test_loss > self.best_loss:
                self.early_stop_counter+=1
                if self.early_stop_counter >= early_stop_epochs:
                    print("Early Stopping!!")
                    break
            else:
                self.best_loss=test_loss
                self.early_stop_counter=0
                torch.save({
                    'epoch':Epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'accuracy': test_acc,
                    'loss': test_loss
                }, './save/model_{:.2f}_epoch{}.pt'.format(test_acc, Epoch))
            