import torch
from typing import Tuple, Optional


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(
    data: torch.tensor, 
    batch_size: int, 
    block_size: int
) -> Tuple[torch.tensor, torch.tensor]:
    """Generate a batch of data of inputs x and targets y

    Args:
        data (torch.tensor): input data
        batch_size (int): number of samples per batch
        block_size (int): number of tokens per sample/sequence

    Returns:
        torch.tensor: batch of data
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    train_data: torch.tensor, 
    val_data: torch.tensor, 
    model: Optional[torch.nn.Module],
    batch_size: int, 
    block_size: int,
    eval_iters: int
):
    """Compute train and val loss for a given torch model
    at some training epoch.

    PD: model.train() tells your model that you are training the model. 
    This helps inform layers such as Dropout and BatchNorm, which 
    are designed to behave differently during training and evaluation.

    Args:
        train_data (torch.tensor): traininig data
        val_data (torch.tensor): validation data
        model (torch.nn.Module): pytorch model
        batch_size (int): number of samples per batch
        block_size (int): number of tokens per sample/sequence
        eval_iters (int): number of batches to evaluate

    Returns:
        dict: train and validation loss
    """
    out = {}
    model.eval()
    iter_dict = {
        "train": train_data,
        "val": val_data
    }
    for split, data in iter_dict.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
