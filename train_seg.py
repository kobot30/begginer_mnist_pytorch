
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm


# セグメンテーション学習
def train_seg_run(n_epochs, model, train_loader, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    total_loss = 0.0
    model.train()

    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            pbar.set_description("[Epoch {} / {}]".format((epoch + 1), n_epochs))
            
            for batch_index, (x_data, y_data) in enumerate(train_loader):
                x = x_data.to(device)
                y = y_data.to(device)             
                optimizer.zero_grad()
 
                pred = model(x)
                loss = calc_loss(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
               
            total_loss = total_loss / batch_index
            pbar.set_postfix(train_loss=total_loss)

    return model


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def calc_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = nn.Sigmoid()(pred)
    
    dice = dice_loss(pred, target)    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss
