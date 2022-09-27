import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from my_dataloader import get_dataloader
from models import model_dict
import os
from utils import AverageMeter, accuracy
import numpy as np
from datetime import datetime

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_names", type=str, nargs="+", default=["resnet20", "resnet20"]
)

parser.add_argument("--root", type=str, default="./dataset")
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar100",
    choices=["cifar100", "cifar10", "CUB", "tinyimagenet"],
    help="dataset",
)
parser.add_argument("--classes_num", type=int, default=100)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epoch", type=int, default=240)
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--milestones", type=int, nargs="+", default=[150, 180, 210])


parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--print_freq", type=int, default=100)
parser.add_argument("--aug_nums", type=int, default=2)
parser.add_argument("--exp_postfix", type=str, default="base")

args = parser.parse_args()
args.num_branch = len(args.model_names)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

exp_name = "_".join(args.model_names) + args.exp_postfix
exp_path = "./baseline/{}/{}".format(args.dataset, exp_name)
os.makedirs(exp_path, exist_ok=True)
print(exp_path)


def train_one_epoch(models, optimizers, train_loader):
    acc_recorder_list = []
    loss_recorder_list = []
    for model in models:
        model.train()
        acc_recorder_list.append(AverageMeter())
        loss_recorder_list.append(AverageMeter())

    for i, (imgs, label) in enumerate(train_loader):
        out_list = []
        teacher_outs = []
        # forward
        for model_idx, model in enumerate(models):

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()

            out = model.forward(imgs[:, model_idx, ...])
            # outputs[model_idx, ...] = out
            out_list.append(out)

        for model_idx, model in enumerate(models):
            loss = F.cross_entropy(out_list[model_idx], label)
            optimizers[model_idx].zero_grad()
            loss.backward()
            optimizers[model_idx].step()

            loss_recorder_list[model_idx].update(loss.item(), n=imgs.size(0))
            acc = accuracy(out_list[model_idx], label)[0]
            acc_recorder_list[model_idx].update(acc.item(), n=imgs.size(0))

    losses = [recorder.avg for recorder in loss_recorder_list]
    acces = [recorder.avg for recorder in acc_recorder_list]
    return losses, acces


def evaluation(models, val_loader):
    acc_recorder_list = []
    loss_recorder_list = []
    for model in models:
        model.eval()
        acc_recorder_list.append(AverageMeter())
        loss_recorder_list.append(AverageMeter())

    with torch.no_grad():
        for img, label in val_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            for model_idx, model in enumerate(models):
                out = model(img)
                acc = accuracy(out, label)[0]
                loss = F.cross_entropy(out, label)
                acc_recorder_list[model_idx].update(acc.item(), img.size(0))
                loss_recorder_list[model_idx].update(loss.item(), img.size(0))
    losses = [recorder.avg for recorder in loss_recorder_list]
    acces = [recorder.avg for recorder in acc_recorder_list]
    return losses, acces


def train(model_list, optimizer_list, train_loader, scheduler_list):
    best_acc = [-1 for _ in range(args.num_branch)]

    f_writers = []
    for i in range(len(model_list)):
        f = open(os.path.join(exp_path, "log_{}test.txt".format(i)), "w")
        f_writers.append(f)

    for epoch in range(args.epoch):
        train_losses, train_acces = train_one_epoch(
            model_list, optimizer_list, train_loader
        )
        val_losses, val_acces = evaluation(model_list, val_loader)

        for i in range(len(best_acc)):
            if val_acces[i] > best_acc[i]:
                best_acc[i] = val_acces[i]
                state_dict = dict(
                    epoch=epoch + 1, model=model_list[i].state_dict(), acc=val_acces[i]
                )
                name = os.path.join(
                    exp_path, args.model_names[i], "ckpt", "best{}.pth".format(i)
                )
                os.makedirs(os.path.dirname(name), exist_ok=True)
                torch.save(state_dict, name)

            scheduler_list[i].step()

        if (epoch + 1) % args.print_freq == 0:
            for j in range(len(best_acc)):
                msg = "epoch:{} model:{} train loss:{:.2f} acc:{:.2f}  val loss{:.2f} acc:{:.2f}\n".format(
                    epoch,
                    args.model_names[j],
                    train_losses[j],
                    train_acces[j],
                    val_losses[j],
                    val_acces[j],
                )
                print(msg)
                f_writers[j].write(msg)
                f_writers[j].flush()

    for k in range(len(best_acc)):
        msg_best = "model:{} best acc:{:.2f}".format(args.model_names[k], best_acc[k])
        print(msg_best)
        f_writers[k].write(msg_best)
        f_writers[k].close()


if __name__ == "__main__":
    train_loader, val_loader = get_dataloader(args)
    model_list = []
    optimizer_list = []
    scheduler_list = []
    for name in args.model_names:
        lr = args.lr
        print(name)
        model = model_dict[name](num_classes=args.classes_num)
        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=args.weight_decay,
        )
        scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)
        model_list.append(model)
        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)

    train(model_list, optimizer_list, train_loader, scheduler_list)
