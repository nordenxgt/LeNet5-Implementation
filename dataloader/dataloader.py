from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def mnist_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = datasets.MNIST(root="./data/", train=False, download=True, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    return train_dataloader, test_dataloader