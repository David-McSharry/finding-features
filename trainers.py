import torch
import torch.nn as nn
import torch.optim as optim


def train_mlp(
        config,
        train_loader,
        mlp,    
    ):
    criterion = torch.nn.CrossEntropyLoss()

    # pretty sure re initing an optim is fine in case we want to continue training an MLP
    # but maybe not
    mlp_optim = torch.optim.Adam(
        mlp.parameters(),
        lr=config['lr'],
        weight_decay=config['mlp_weight_decay']
    )

    for epoch in range(config['epochs']):
        for i, (X, y) in enumerate(train_loader):
            X = X.float()
            y = y.long()
            y_pred = mlp(X)
            loss = criterion(y_pred, y)
            mlp_optim.zero_grad()
            loss.backward()
            mlp_optim.step()
            if i % 10 == 0:
                # accu
                correct = (y_pred.argmax(dim=1) == y).sum().item()
                total = y.size(0)
                print(f'Epoch {epoch}, iteration {i}, loss: {loss.item()}, accuracy: {correct / total}')
    return mlp


def test_mlp(
        config,
        test_loader,
        mlp,
    ):

    total = 0
    correct = 0
    for (X, y) in test_loader:
        X = X.float()
        y = y.long()
        y_pred = mlp(X)
        correct += (y_pred.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    print(f'Test accuracy: {correct / total}')
    return None