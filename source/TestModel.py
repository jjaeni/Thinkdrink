import os, random
import numpy as np
import torch
from torchvision import transforms, datasets

from dataset import StarbucksDataset
from model import ResNet50V2CustomLayer
from trainer import ResNet50V2Trainer

# 성능 재현을 위한 seed 고정
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.bencmark=False
    os.environ['PYTHONHASHSEED']=str(seed)
seed_everything(2024)

dataset = StarbucksDataset()
train_loader, val_loader, test_loader = dataset.splitdata()

model = ResNet50V2CustomLayer()
checkpoint=torch.load('./saved/model_9_80.30.pt', map_location=torch.device('cpu')) # 모델은 GPU 이용하여 학습시켰기 때문에 Location 이용하여 CPU로 전환
model.load_state_dict(checkpoint.get('model_state_dict'))

trainer = ResNet50V2Trainer()
test_loss, test_acc = trainer.evaluate(model, val_loader)
print('Test Accuracy: {:.2f}% \t Test Loss: {:.4f}'.format(test_acc, test_loss))