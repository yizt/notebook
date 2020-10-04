# -*- coding: utf-8 -*-
"""
 @File    : art.py
 @Time    : 2020/10/4 下午4:21
 @Author  : yizuotian
 @Description    :
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm

trans = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class DataSetArt(Dataset):
    def __init__(self, data_root, img_transforms=None,
                 target_transforms=None, mode='train', **kwargs):

        self.mode = mode
        self.path = os.path.join(data_root, self.mode)
        self.train_gt_path = os.path.join(data_root, 'train.csv')
        self.gt_list = pd.read_csv(self.train_gt_path)['label'].values
        self.image_path_list = [os.path.join(self.path, file_name)
                                for file_name in os.listdir(self.path)]

        self.transforms = img_transforms
        self.target_transforms = target_transforms
        super(DataSetArt, self).__init__()

    def __getitem__(self, item):
        im_path = self.image_path_list[item]
        im = Image.open(im_path)
        im = im.convert('RGB')

        if self.transforms:
            im = self.transforms(im)

        if self.mode == 'test':
            return {'image': im}
        # train
        index = int(os.path.splitext(os.path.basename(im_path))[0])  # eg: /path/to/img/13.img
        gt = self.gt_list[index]
        if self.target_transforms:
            label = self.target_transforms(np.array(gt))
        return {'image': im,
                'target': label}

    def __len__(self):
        return len(self.image_path_list)


class BaseModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self.num_classes = num_classes

        # 使用预训练基模型
        self.cnn = self.feature_extractor()
        for name, value in self.cnn.named_parameters():
            value.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 分类
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """

        :param x: [B,C,H,W]
        :return:
        """
        x = self.cnn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @classmethod
    def feature_extractor(cls):
        return nn.Identity()


class ResNetModel(BaseModel):
    @classmethod
    def feature_extractor(cls):
        resnet = models.resnet18(pretrained=True)
        return nn.Sequential(*list(resnet.children())[:-2])


def train(args):
    torch.backends.cudnn.benchmark = True

    data_set = DataSetArt(args.data_root,
                          img_transforms=transforms.Compose([
                              transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                          ]),
                          target_transforms=transforms.Lambda(lambda x: torch.from_numpy(x)))
    train_sampler = torch.utils.data.RandomSampler(data_set)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, sampler=train_sampler,
                             num_workers=args.workers)

    #
    net = ResNetModel(num_classes=49)
    params = filter(lambda p: p.requires_grad, net.parameters())
    # for n, p in net.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    print(net)
    net.train()
    net.to(device)

    optimizer = optim.Adadelta(params, weight_decay=args.weight_decay)

    # 加载预训练模型
    if args.init_epoch > 0:
        checkpoint = torch.load(os.path.join(args.output_dir,
                                             'art.{:03d}.pth'.format(args.init_epoch)),
                                map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        net.load_state_dict(checkpoint['model'])

    # 训练
    for epoch in range(args.init_epoch, args.epochs):
        epoch_loss = 0
        accuracy_num = 0
        for sample in tqdm(data_loader):
            image = sample['image'].to(device)
            target = sample['target'].to(device)

            outputs = net(image)  # [B,N,C]
            loss = F.cross_entropy(outputs, target)
            # 梯度更新
            net.zero_grad()
            loss.backward()
            optimizer.step()
            # 当前轮的loss
            epoch_loss += loss.item() * image.size(0)
            # 统计精度
            _, class_ids = torch.max(outputs, dim=-1)
            accuracy_num += np.sum(class_ids.cpu().detach().numpy() == sample['target'].numpy())

        epoch_loss = epoch_loss / len(data_loader.dataset)
        acc = accuracy_num / len(data_loader.dataset)
        # 打印日志,保存权重
        print('Epoch: {}/{} loss: {:03f} acc: {:.3f}'.format(epoch + 1,
                                                             args.epochs,
                                                             epoch_loss,
                                                             acc))

        # 保存模型
        if args.output_dir:
            checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch + 1,
                'args': args}
            torch.save(checkpoint,
                       os.path.join(args.output_dir, 'art.{:03d}.pth'.format(epoch + 1)))

    return net


def inference(args, net):
    net.eval()
    data_set = DataSetArt(args.data_root,
                          img_transforms=trans,
                          target_transforms=transforms.Lambda(lambda x: torch.from_numpy(x)),
                          mode='test')
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.workers)
    class_id_list = []
    for sample in tqdm(data_loader):
        image = sample['image'].to(device)
        outputs = net(image)  # [B,N,C]
        _, class_ids = torch.max(outputs, dim=-1)
        class_id_list.append(class_ids.cpu().detach().numpy())

    class_id_np = np.concatenate(class_id_list, axis=0)
    class_id_pd = pd.DataFrame(class_id_np)
    class_id_pd.to_csv('rst.art.csv', header=None)


if __name__ == '__main__':
    """
    Usage:
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export CUDA_VISIBLE_DEVICES=0
    python demo2.py -i /home/mydir/dataset/71_OCR_2 --device cuda --epochs 193 --init-epoch 193
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data-root', type=str, default='./Art')
    parser.add_argument("--device", type=str, default='cpu', help="cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="epochs")
    parser.add_argument("--init-epoch", type=int, default=0, help="init epoch")
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight decay (default: 0)')
    parser.add_argument("--workers", type=int, default=4, help="number of workers")
    parser.add_argument('--output-dir', default='./output', help='path where to save')
    # test_model()
    arguments = parser.parse_args(sys.argv[1:])
    device = torch.device(
        'cuda' if arguments.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    net = train(arguments)

    inference(arguments, net)
