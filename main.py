# -*- coding: utf-8 -*-
import argparse
import time

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from model import E2EModel
from trainer import BC_trainer
import os

from dataset.multidemodataset import HabitatDemoMultiGoalDataset
from torch.utils.data import DataLoader
from configs.default import get_config
import warnings
from main_util import make_print_to_file

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default="./configs/vgm.yaml",
    type=str,
    required=False,
    help="path to config yaml containing info about experiment",
)
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    help="gpus",
)
parser.add_argument("--version", type=str, default="test", help="name to save")
parser.add_argument(
    "--stop",
    action="store_true",
    default=True,
    help="include stop action or not",
)
parser.add_argument(
    "--data-dir",
    default="your_dataset_dir",
    type=str,
)  # your_data_path
parser.add_argument("--resume", default="none", type=str)
parser.add_argument("--difficulty", default="easy", type=str)
parser.add_argument("--actions", default=3, type=int)
args = parser.parse_args()


warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(
    threshold=np.inf
)  # threshold specifies when to use ellipsis; np.inf means never truncate


"""create path for model and summary"""
date = str(datetime.today())[:10]
model_path = "./record/" + date + "/models/"
summary_path = "./record/" + date + "/summary/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(summary_path):
    os.makedirs(summary_path)

"""parameters model instantiation"""
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
action_space = args.actions

model = E2EModel(action_space=action_space, device=device, batch_size=4)
if torch.cuda.is_available():
    model = model.cuda()


def train(difficulty="easy"):
    print("Training Start")
    model.train()  # model.eval()
    writer = SummaryWriter(summary_path)

    config = get_config(args.config)

    config.defrost()
    config.NUM_PROCESSES = config.BC.batch_size
    config.TORCH_GPU_ID = args.gpu
    config.freeze()

    trainer = BC_trainer(model, device=device)

    DATA_DIR = args.data_dir

    train_data_list = [
        os.path.join(DATA_DIR, "train", difficulty, x)
        for x in sorted(os.listdir(os.path.join(DATA_DIR, "train", difficulty)))
    ]
    train_data_list += [
        os.path.join(DATA_DIR, "train", "medium", x)
        for x in sorted(os.listdir(os.path.join(DATA_DIR, "train", "medium")))
    ]
    np.random.shuffle(train_data_list)
    params = {
        "batch_size": config.BC.batch_size,
        "shuffle": True,
        "num_workers": config.BC.num_workers,
        "pin_memory": True,
    }
    train_dataset = HabitatDemoMultiGoalDataset(config, train_data_list, args.stop)

    version_name = config.saving.name if args.version == "none" else args.version
    version_name = version_name
    version_name += "_start_time:{}".format(time.ctime())

    start_step = 0
    start_epoch = 0
    if args.resume != "none":  # Resume training from a checkpoint
        sd = torch.load(args.resume)
        start_epoch, start_step = sd["trained"]
        trainer.agent.load_state_dict(sd["state_dict"])
        print(
            "load {}, start_ep {}, strat_step {}".format(
                args.resume, start_epoch, start_step
            )
        )

    print_every = 5
    save_every = config.saving.save_interval
    eval_every = config.saving.eval_interval

    start = time.time()
    temp = start
    step = start_step
    step_values = [10000, 50000, 100000]
    step_index = 0
    lr = config.BC.lr

    def adjust_learning_rate(optimizer, step_index, lr_decay):
        lr = config.BC.lr * (lr_decay**step_index)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    trainer.to(device)
    trainer.train()
    for epoch in range(start_epoch, config.BC.max_epoch):
        train_dataloader = DataLoader(train_dataset, **params)
        train_iter = iter(train_dataloader)
        loss_summary_dict = {}
        for batch in train_iter:
            results, loss_dict = trainer(batch)
            for k, v in loss_dict.items():
                if k not in loss_summary_dict.keys():
                    loss_summary_dict[k] = []
                loss_summary_dict[k].append(v)

            if step in step_values:
                step_index += 1
                lr = adjust_learning_rate(trainer.optim, step_index, 0.1)

            if step % print_every == 0:
                loss_str = ""
                writer_dict = {}
                for k, v in loss_summary_dict.items():
                    value = np.array(v).mean()
                    loss_str += "%s: %.3f " % (k, value)
                    writer_dict[k] = value
                writer_dict["node_num"] = results["node_num"]
                print(
                    "time = %.2fm, epo %d, step %d, lr: %.5f, %ds per %d iters || loss : "
                    % (
                        (time.time() - start) // 60,
                        epoch + 1,
                        step + 1,
                        lr,
                        time.time() - temp,
                        print_every,
                    ),
                    loss_str,
                )
                loss_summary_dict = {}
                temp = time.time()
                writer.add_scalars("loss", writer_dict, step)

            if step % save_every == 0:
                trainer.save(
                    file_name=os.path.join(
                        model_path, "epoch%04diter%05d.pt" % (epoch, step)
                    ),
                    epoch=epoch,
                    step=step,
                )

            step += 1
    print("===> end training")


if __name__ == "__main__":
    make_print_to_file()
    train(difficulty="easy")
