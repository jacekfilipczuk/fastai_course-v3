import warnings
import torch

warnings.filterwarnings("ignore")

def check_if_using_GPU():
    print('Id of the currently used device: ', torch.cuda.current_device())
    print('Name of the currently used device: ', torch.cuda.get_device_name(0))
    print('Number of available devices (CPU, GPU): ', torch.cuda.device_count())
    print('Check if cuda is available: ', torch.cuda.is_available())
    

