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
import random

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--model_names", type=str, default="vgg16")


parser.add_argument("--root", type=str, default="./dataset")
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--classes_num", type=int, default=100)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar100",
    choices=["cifar100", "cifar10", "CUB", "tinyimagenet"],
    help="dataset",
)

parser.add_argument("--T", type=float)
parser.add_argument("--alpha", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epoch", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--milestones", type=int, nargs="+")

parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--gamma", type=float, default=0.1)

parser.add_argument("--seed", type=int, default=95)
parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--print_freq", type=int, default=100)
parser.add_argument("--aug_nums", type=int, default=2)  #
parser.add_argument("--exp_postfix", type=str, default="TP3_0.5")  #

args = parser.parse_args()
args.num_branch = len(args.model_names)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

exp_name = "_".join(args.model_names) + args.exp_postfix
exp_path = "./dlb/{}/{}".format(args.dataset, exp_name)
os.makedirs(exp_path, exist_ok=True)
print(exp_path)


def train_one_epoch(model, optimizer, train_loader, alpha, pre_data, pre_out):
    model.train()
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()

    for i, data in enumerate(train_loader):

        imgs, label = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            label = label.cuda()
        out = model.forward(imgs[:, 0, ...])

        if pre_data != None:
            pre_images, pre_label = pre_data
            if torch.cuda.is_available():
                pre_images = pre_images.cuda()
                pre_label = pre_label.cuda()
            out_pre = model.forward(pre_images[:, 1, ...])
            ce_loss = F.cross_entropy(
                torch.cat((out_pre, out), dim=0), torch.cat((pre_label, label), dim=0)
            )  #
            dml_loss = (
                F.kl_div(
                    F.log_softmax(out_pre / args.T, dim=1),
                    F.softmax(pre_out.detach() / args.T, dim=1),  # detach
                    reduction="batchmean",
                )
                * args.T
                * args.T
            )
            loss = ce_loss + alpha * dml_loss
        else:
            loss = F.cross_entropy(out, label)

        loss_recorder.update(loss.item(), n=imgs.size(0))
        acc = accuracy(out, label)[0]
        acc_recorder.update(acc.item(), n=imgs.size(0))

        pre_data = data
        pre_out = out

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses = loss_recorder.avg
    acces = acc_recorder.avg

    return losses, acces, pre_data, pre_out


def evaluation(model, val_loader):
    model.eval()
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()

    with torch.no_grad():
        for img, label in val_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            out = model(img)
            acc = accuracy(out, label)[0]
            loss = F.cross_entropy(out, label)
            acc_recorder.update(acc.item(), img.size(0))
            loss_recorder.update(loss.item(), img.size(0))
    losses = loss_recorder.avg
    acces = acc_recorder.avg
    return losses, acces


def train(model, optimizer, train_loader, scheduler):
    best_acc = -1

    f = open(os.path.join(exp_path, "log_test.txt"), "w")

    pre_data, pre_out = None, None

    for epoch in range(args.epoch):
        alpha = args.alpha
        train_losses, train_acces, pre_data, pre_out = train_one_epoch(
            model, optimizer, train_loader, alpha, pre_data, pre_out
        )
        val_losses, val_acces = evaluation(model, val_loader)

        if val_acces > best_acc:
            best_acc = val_acces
            state_dict = dict(epoch=epoch + 1, model=model.state_dict(), acc=val_acces)
            name = os.path.join(exp_path, args.model_names, "ckpt", "best.pth")
            os.makedirs(os.path.dirname(name), exist_ok=True)
            torch.save(state_dict, name)

        scheduler.step()

        if (epoch + 1) % args.print_freq == 0:
            msg = "epoch:{} model:{} train loss:{:.2f} acc:{:.2f}  val loss{:.2f} acc:{:.2f}\n".format(
                epoch,
                args.model_names,
                train_losses,
                train_acces,
                val_losses,
                val_acces,
            )
            print(msg)
            f.write(msg)
            f.flush()

    msg_best = "model:{} best acc:{:.2f}".format(args.model_names, best_acc)
    print(msg_best)
    f.write(msg_best)
    f.close()


if __name__ == "__main__":
    train_loader, val_loader = get_dataloader(args)
    lr = args.lr
    model = model_dict[args.model_names](num_classes=args.classes_num)
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

    train(model, optimizer, train_loader, scheduler)
