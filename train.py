import torch
import torch.nn as nn
import HELPER
from model import create_convolutional_solver_instance




torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

HELPER.reset_seed(0)
data_dict = HELPER.data.preprocess_cifar10(cuda=True, dtype=torch.float64, flatten=False)
solver = create_convolutional_solver_instance(data_dict, torch.float32, 'cuda')

solver.train()

torch.backends.cudnn.benchmark = False

print('Validation set accuracy: ', solver.check_accuracy(data_dict['X_val'], data_dict['y_val']))




model_path=r"C:\Users\19017\Desktop\DaChuang\model.pth"
torch.save(solver.model, model_path)