from letsformer import TransformerLM
from letsformer.debug import console, DEBUG, LossMonitor
from letsformer.optim import AdamW
from letsformer.functions import cross_entropy_loss, cosine_schedule, gradient_clipping
import torch
from typing import IO, BinaryIO
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import os
import time

def data_loader(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device,
)-> tuple[torch.Tensor, torch.Tensor]:
    '''
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    '''
    # 从数据集中随机选择起始位置
    indices = np.random.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,)
    )
    # 根据起始位置获取输入序列和对应的标签
    inputs = np.stack([dataset[index:index+context_length] for index in indices])
    labels = np.stack([dataset[index+1:index+context_length+1] for index in indices])
    
    # 将numpy数组转换为torch张量，并移动到指定设备
    return (
        torch.from_numpy(inputs).long().to(device),
        torch.from_numpy(labels).long().to(device)
    )

def save_checkpoint(
	model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    # 准备好要保存的文件
    if isinstance(out, str) or isinstance(out, os.PathLike):
        out = open(out, 'wb')

    # 保存模型状态
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration,
        },
        out
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    # 加载模型状态
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration']

def read_memmap_data(train_data_path: str | os.PathLike):
    '''
    读取训练数据集。
    '''
    return np.memmap(
        train_data_path,
        dtype=np.int32,
        mode="r",
    )

def val_iterator(memmap_arr, batch_size, context_length):
    N = len(memmap_arr)
    nb = (N-context_length-1)//batch_size
    for bi in range(nb):
        base = bi*batch_size
        x = np.stack([memmap_arr[i:i+context_length] for i in range(base, base+batch_size)])
        y = np.stack([memmap_arr[i+1:i+context_length+1] for i in range(base, base+batch_size)])
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train_lm(model_params: dict):
    # 1. 准备训练参数
    train_data_path = model_params["train_data_path"]
    valid_data_path = model_params["valid_data_path"]

    model_weights_path = model_params.get("model_weights_path", None)
    save_model_dir = model_params["save_model_dir"]
    
    device = torch.device("cuda")
    # device = torch.device("cpu")
    epochs: int = model_params["epochs"]
    valid_interval: int = model_params["valid_interval"]
    lr: float = model_params["lr"]
    betas: tuple[float, float] = model_params["betas"]
    weight_decay: float = model_params["weight_decay"]
    
    console.print("Training data path:", train_data_path)
    console.print("Validation data path:", valid_data_path)
    if model_weights_path is not None:
        console.print("Using model weights path:", model_weights_path)
    console.print("Save model dir:", save_model_dir)
    console.print("Epochs:", epochs)
    console.print("Valid interval:", valid_interval)
    console.print("Device:", device)
    console.print("lr:", lr)
    console.print("betas:", betas)
    console.print("weight_decay:", weight_decay)
    
    vocab_size: int = model_params["vocab_size"]
    context_length: int = model_params["context_length"]
    d_model: int = model_params["d_model"]
    num_layers: int = model_params["num_layers"]
    num_heads: int = model_params["num_heads"]
    d_ff: int = model_params["d_ff"]
    rope_theta: float = model_params["rope_theta"]
    max_seq_len: int = model_params["max_seq_len"]
    batch_size: int = model_params["batch_size"]

    console.print("vocab_size:", vocab_size)
    console.print("context_length:", context_length)
    console.print("d_model:", d_model)
    console.print("num_layers:", num_layers)
    console.print("num_heads:", num_heads)
    console.print("d_ff:", d_ff)
    console.print("rope_theta:", rope_theta)
    console.print("max_seq_len:", max_seq_len)
    console.print("batch_size:", batch_size)

    # 2. 加载数据集
    train_data_tokens = read_memmap_data(train_data_path)    # 一维数组
    console.print("train_data_tokens shape:", train_data_tokens.shape)

    valid_data_tokens = read_memmap_data(valid_data_path)    # 一维数组
    console.print("valid_data_tokens shape:", valid_data_tokens.shape)

    # 3. 创建模型和优化器
    model = TransformerLM(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
        max_seq_len, batch_size,
        device=device,
    ).to(device)

    if model_weights_path is not None:
        model.load_state_dict(torch.load(model_weights_path))

    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, eps=1e-8, weight_decay=weight_decay)

    train_loss_monitor = LossMonitor(title="Train Loss Monitor")
    valid_loss_monitor = LossMonitor(title="Valid Loss Monitor")

    # 4. 训练模型
    start_time = time.time()
    for epoch in range(epochs):
        try:
            # ----------------------------------------
            #                  Train
            # ----------------------------------------
            model.train()
            inputs, targets = data_loader(train_data_tokens, batch_size=batch_size, context_length=context_length, device=device)

            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            gradient_clipping(model.parameters(), 1.0)

            # 学习率调度
            current_lr = cosine_schedule(
                epoch,
                max_learning_rate=0.0001,
                min_learning_rate=0.00001,
                warmup_iters=500,
                cosine_cycle_iters=10000,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.step()

            # 信息输出

            train_loss_monitor.add_loss(epoch, loss.item())

            # 打印信息
            if epoch % 20 == 0:
                console.print(f"Epoch {epoch+1}/{epochs}")
                console.print(f"Loss: {loss.item():.4f}")
                console.print("Epoch completed")
                console.print("="*50)
            
            # ----------------------------------------
            #                Validate
            # ----------------------------------------
            if epoch % valid_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    count = 0
                    for inputs_val, targets_val in val_iterator(valid_data_tokens, batch_size, context_length):
                        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                        val_logits = model(inputs_val)
                        val_loss = cross_entropy_loss(val_logits, targets_val)
                        # 统计 val_loss
                        val_losses.append(val_loss.item())
                        count += 1
                        if count >= 10:
                            break
                    val_loss_mean = np.mean(val_losses)
                    valid_loss_monitor.add_loss(epoch, val_loss_mean)
                    console.print(f"VALID mean loss: {val_loss_mean:.4f}")


        except AssertionError as e:
            console.print(f"AssertionError in epoch {epoch+1}: {e}")
            console.print(f"Max token ID: {inputs.max()}")
            console.print(f"Max token ID: {targets.max()}")
            # 保存 checkpoint
            save_checkpoint(model, optimizer, epoch, os.path.join(save_model_dir, "checkpoint.pt"))
            console.print(f"Checkpoint saved at epoch {epoch+1}")
            console.print("="*50)
            return
        except Exception as e:
            console.print(f"Error in epoch {epoch+1}: {e}")
            # 保存 checkpoint
            save_checkpoint(model, optimizer, epoch, os.path.join(save_model_dir, "checkpoint.pt"))
            console.print(f"Checkpoint saved at epoch {epoch+1}")
            console.print("="*50)
            return
        
    end_time = time.time()
    console.print(f"Training time: {end_time - start_time:.2f} seconds")
    
    # 5. 保存模型
    torch.save(model.state_dict(), os.path.join(save_model_dir, "model_new.pt"))

    # 6. 绘制loss曲线
    train_loss_monitor.finalize(save_path=os.path.join(save_model_dir, "train_loss_curve.png"))
    valid_loss_monitor.finalize(save_path=os.path.join(save_model_dir, "valid_loss_curve.png"))



if __name__ == "__main__":
    model_params = {
        "train_data_path": "./data/tinystories_train.npy",
        "valid_data_path": "./data/tinystories_valid.npy",
        # "model_weights_path": "",
        "save_model_dir": "./data/model/",
        "epochs": 6000,
        "valid_interval": 100,
        "lr": 0.0002,
        "betas": (0.9, 0.999),
        "weight_decay": 0.01,

        "vocab_size": 10000,
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10000,
        "max_seq_len": 256,
        "batch_size": 32,
    }
    train_lm(model_params)