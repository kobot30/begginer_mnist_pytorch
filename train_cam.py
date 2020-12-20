from tqdm import tqdm
import torch
import torch.nn as nn


def train_cam_run(n_epochs, model, train_loader, test_loader, device):
    loss_func = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            pbar.set_description("[Epoch {} / {}]".format((epoch + 1), n_epochs))

            train_loss = train(model, train_loader, device, optimizer, loss_func)
            test_accuracy = test(model, test_loader, device)

            pbar.set_postfix(train_loss=train_loss, test_acc=test_accuracy[0])


def train(model, train_loader, device, optimizer, loss_func):
    model.train()
    for _, [x_data, y_data] in enumerate(train_loader):
        x, y = x_data.to(device), y_data.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()

    return loss.item()


def test(model, test_loader, device):
    top_1_count = torch.FloatTensor([0])
    total = torch.FloatTensor([0])
    model.eval()
    with torch.no_grad():
        for x_data, y_data in test_loader:
            x, y = x_data.to(device), y_data.to(device)
            output = model(x)

            values, idx = output.max(dim=1)
            top_1_count += torch.sum(y == idx).float().cpu().data
            total += y_data.size(0)

    return (top_1_count / total).numpy()
