from tqdm import tqdm
from torch import nn, optim

from model_select import ModelSelectFlag
from test import test


def train(model, train_loader, optimizer, criterion, device, flag=None):
    total_loss = 0.0
    model.train()

    for batch_index, (x_data, y_data) in enumerate(train_loader):
        x, y = x_data.to(device), y_data.to(device)
        optimizer.zero_grad()
        pred_y = model(x)

        if flag == ModelSelectFlag.REGRESSION:
            pred_y = pred_y[:, 0]  # shape = [batch, 1] -> [batch]
            loss = criterion(pred_y, y.float())  # cast float
        elif flag == ModelSelectFlag.CLASSIFICATION:
            loss = criterion(pred_y, y)
        elif flag == ModelSelectFlag.AUTOENCODER:
            loss = criterion(x, pred_y)  # 予測データと入力データを比較
        else:
            raise RuntimeError("loss empty")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / batch_index


def train_run(n_epochs, train_loader, test_loader, model, device, flag=None):
    # loss function
    if flag == ModelSelectFlag.REGRESSION or flag == ModelSelectFlag.AUTOENCODER:
        criterion = nn.MSELoss()
    elif flag == ModelSelectFlag.CLASSIFICATION:
        criterion = nn.CrossEntropyLoss()
    else:
        raise RuntimeError("Model flag error !")

    # optimizer
    optimizer_cls = optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []
    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            pbar.set_description("[Epoch {} / {}]".format((epoch + 1), n_epochs))
            train_loss = train(model, train_loader, optimizer, criterion, device, flag)
            test_loss = test(model, test_loader, optimizer, criterion, device, flag)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            pbar.set_postfix(train_loss=train_loss, test_loss=test_loss)

    return train_losses, test_losses
