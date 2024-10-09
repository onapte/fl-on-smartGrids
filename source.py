import torch
import numpy as np

def generate_random_tensor(shape):
    return torch.rand(shape)

def normalize_tensor(tensor):
    return (tensor - tensor.mean()) / tensor.std()

def compute_average(tensors):
    return sum(tensors) / len(tensors)

def log_message(message):
    with open('log.txt', 'a') as f:
        f.write(f"{message}\n")

def create_random_data(num_samples=100, num_features=10):
    return np.random.rand(num_samples, num_features)

def evaluate_model_performance(y_true, y_pred):
    accuracy = (y_true == y_pred).mean()
    return accuracy
