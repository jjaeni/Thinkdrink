import numpy as np
import random, os, torch
from source.setup import setup_config
from source.model import ResNet50V2CustomLayer
from source.trainer import ResNet50V2Trainer

class ModelTest:
    def __init__(self):
        if torch.cuda.is_available():
            self.DEVICE = torch.device('cuda')
        else:
            self.DEVICE = torch.device('cpu')
        self.model = ResNet50V2CustomLayer().to(self.DEVICE)
        self.trainer = ResNet50V2Trainer()
        self.seed = setup_config()['SEED']

    def seed_all(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.bencmark=False
        os.environ['PYTHONHASHSEED']=str(self.seed)

    def progress(self):
        self.trainer.train_model(self.model, early_stop_epochs=5)

    def check_performance(self):
        #self.seed_all()
        checkpoint = torch.load('./saved/model_9_80.30.pt', map_location=self.DEVICE)
        self.model.load_state_dict(checkpoint.get('model_state_dict'))

        test_loss, test_acc = self.trainer.evaluate(self.model)
        print('Test Accuracy: {:.2f}% \t Test Loss: {:.4f}'.format(test_acc, test_loss))

if __name__=='__main__':
    #ModelTest().progress()
    ModelTest().check_performance()