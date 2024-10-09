import torch
from torchvision import datasets, transforms

def get_data_loader(batch_size=32, dataset='MNIST'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if dataset == 'MNIST':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def get_test_data_loader(batch_size=32, dataset='MNIST'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if dataset == 'MNIST':
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'CIFAR10':
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def save_model(model, filename='model.pth'):
    torch.save(model.state_dict(), filename)

def load_model(model, filename='model.pth'):
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model
