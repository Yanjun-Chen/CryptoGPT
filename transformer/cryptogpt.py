import os
import sys
import json

import torch
from torch.utils.data import Dataset

from cryptogpt.model import GPT
from cryptogpt.trainer import Trainer
from cryptogpt.utils import set_seed, setup_logging, CfgNode as CN

import pandas as pd
import wandb

# -----------------------------------------------------------------------------


def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/stockgpt'

    # data
    C.data = CryptoDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    # gpt-mini settings
    C.model.n_layer = 6
    C.model.n_query_head = 6
    C.model.n_kv_head = 6
    C.model.n_embd = 192
    C.model.rope = False  # toggle True or False to turn rope on and off respectively

    # trainer
    C.trainer = Trainer.get_default_config()
    # C.trainer.learning_rate = 5e-6 # the model we're using is so small that we can go a bit faster
    # the model we're using is so small that we can go a bit faster
    C.trainer.learning_rate = 5e-3

    return C

# -----------------------------------------------------------------------------


class CryptoDataset(Dataset):
    """
    PyTorch dataset for stock data
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, dataset):
        self.config = config
        self.data = dataset

        # self.feature = ['Open', 'High', 'Low', 'Close']
        self.feature = ['open', 'high', 'low', 'close']
        self.word_size = len(self.feature)

        print('data has %d items. each has %d values.' %
              (len(self.data), self.word_size))

    def get_word_size(self):
        return self.word_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # print(idx)
        # grab a chunk of (block_size + 1) rows from the data
        chunk = self.data.iloc[idx:idx + self.config.block_size + 1]
        # extract 'Open', 'High', 'Low', 'Close', and 'Volume' values
        x = torch.tensor(chunk[self.feature].values[:-1], dtype=torch.float)
        # you can modify this based on your use case
        y = torch.tensor(chunk[self.feature].values[1:], dtype=torch.float)
        return x, y


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # SET UP
    # dataset_name = "BTCUSDT_kline_22_10-24_14"
    # dataset_name = "BTCUSDT_kline_1h_22_10-24_14"
    dataset_name = "ETHUSDT_kline_1h_22_04-24_04"
    test_set_len = 24 * 5
    block_size = 128

    # get default config and overrides from the command line, if any
    config = get_config()
    set_seed(config.system.seed)

    data_path = "../data/" + dataset_name
    wandb_project = "CryptoGPT"
    wandb_run_name = f"{dataset_name}-b:{config.trainer.batch_size}-it:{config.trainer.max_iters}-lr:{config.trainer.learning_rate}"
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    data = pd.read_csv(data_path)
    train_set = data.iloc[:-(3 * test_set_len) + block_size]
    val_set = data.iloc[-(3 * test_set_len) - block_size: - test_set_len]
    test_set = data.iloc[- test_set_len - block_size:]
    train_dataset = CryptoDataset(config.data, train_set)
    val_dataset = CryptoDataset(config.data, val_set)
    test_dataset = CryptoDataset(config.data, test_set)

    # construct the model
    config.model.word_size = train_dataset.get_word_size()
    config.model.block_size = train_dataset.get_block_size()
    print(config)
    model = GPT(config.model)

    if config.model.pretrained_folder != None:
        assert os.path.normpath(os.path.abspath(config.model.pretrained_folder)) != os.path.normpath(os.path.abspath(
            config.system.work_dir)), "pretrained folder cannot be same as current folder. Change the folder name of your pretrained model or current directory using flags"
        model.load_pretrained(config.model.pretrained_folder)

    setup_logging(config)

    # some tests
    assert config.model.n_query_head % config.model.n_kv_head == 0, f"query_heads ({config.model.n_query_head}) must be divisible by kv_heads ({config.model.n_kv_head})"
    assert ((config.model.n_embd % config.model.n_query_head == 0) and (config.model.n_embd % config.model.n_kv_head == 0)), f"embed_dim ({config.model.n_embd}) must be divisible by query_heads ({config.model.n_query_head}) and kv_heads ({config.model.n_kv_head})"

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset,
                      val_dataset, test_dataset)

    train_losses = []
    attn_times = []
    attn_mem = []
    # iteration callback

    def batch_end_callback(trainer):
        if trainer.iter_num % 1 == 0:
            train_losses.append(trainer.loss.item())
            attn_times.append(trainer.attn_times*1000)
            if trainer.device == "cuda":
                if trainer.iter_num % 10 == 0:
                    print(f"iter_dt {trainer.iter_dt:.2f}s; iter {trainer.iter_num}: train loss {trainer.loss.item():.5e}; R^2 score {trainer.r2}; attn_times {trainer.attn_times*1000:.2f}ms;mem_consumed {trainer.memory_consumed/(1024*1024):.2f}MB")
                attn_mem.append(trainer.memory_consumed/(1024*1024))
            else:
                if trainer.iter_num % 10 == 0:
                    print(f"iter_dt {trainer.iter_dt:.2f}s; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}; R^2 score {trainer.r2}; attn_times {trainer.attn_times*1000:.2f}ms;mem_consumed - not available on CPU")

            if trainer.iter_num % 100 == 0:
                model.eval()
                with torch.no_grad():
                    eval_iters = len(trainer.val_loader)
                    losses = torch.zeros(eval_iters)
                    r2 = torch.zeros(eval_iters)
                    data_iter = iter(trainer.val_loader)
                    for k in range(eval_iters):

                        batch = next(data_iter)
                        batch = [t.to(trainer.device) for t in batch]
                        x, y = batch

                        # forward the model
                        logits, loss, _, _, r2_ = model(x, y)
                        losses[k] = loss.item()
                        r2[k] = r2_

                    print(f"iter {trainer.iter_num}: Train loss {trainer.loss.item():.5e}; Train R^2 score {trainer.r2}; Val loss {losses.mean():.5e}; Val R^2 score {r2.mean()}")

                    # Log metrics to Wandb
                    wandb.log({
                        "train_loss": trainer.loss.item(),
                        "train_r2_score": trainer.r2,
                        "val_loss": losses.mean().item(),
                        "val_r2_score": r2.mean()
                    }, step=trainer.iter_num)

                    print("saving model")
                    ckpt_path = os.path.join(
                        config.system.work_dir, "model.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    print("saving loss and attention logs")
                    data = {
                        "train_losses": train_losses,
                        "attention_computation_time": attn_times,
                        "attention_computation_memory": attn_mem
                    }
                    with open(os.path.join(config.system.work_dir, 'train_logs.json'), 'w') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                # revert model to training mode
            model.train()

    def epoch_end_callback(trainer):
        model.eval()
        with torch.no_grad():
            eval_iters = len(trainer.test_loader)
            losses = torch.zeros(eval_iters)
            r2 = torch.zeros(eval_iters)
            data_iter = iter(trainer.test_loader)
            for k in range(eval_iters):

                batch = next(data_iter)
                batch = [t.to(trainer.device) for t in batch]
                x, y = batch

                # forward the model
                logits, loss, _, _, r2_ = model(x, y)
                losses[k] = loss.item()
                r2[k] = r2_

            print(f"iter {trainer.iter_num}: Train loss {trainer.loss.item():.5e}; Train R^2 score {trainer.r2}; Test loss {losses.mean():.5e}; Test R^2 score {r2.mean()}")

            # Log metrics to Wandb
            wandb.log({
                # "train_loss": trainer.loss.item(),
                # "train_r2_score": trainer.r2,
                "test_loss": losses.mean().item(),
                "test_r2_score": r2.mean()
            }, step=trainer.iter_num)

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.set_callback('on_epoch_end', epoch_end_callback)

    # run the optimization
    trainer.run()
