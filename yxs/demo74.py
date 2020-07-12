# -*- coding: utf-8 -*-
"""
 @File    : demo74.py
 @Time    : 2020/7/11 下午12:55
 @Author  : yizuotian
 @Description    :
"""
import argparse
import codecs
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm

gender_list = ['Men', 'Women', 'Unisex', 'Boys', 'Girls']
category_list = ['Apparel', 'Accessories', 'Footwear', 'Personal Care',
                 'Free Items', 'Sporting Goods', 'Home']
sport_list = ['Casual', 'Home', 'Sports', 'Ethnic', 'Formal', 'Smart Casual', 'Party', 'Travel']


class YXSOcrDataset(Dataset):
    def __init__(self, data_root, transforms=None,
                 target_transforms=None, mode='train', **kwargs):
        super(YXSOcrDataset, self).__init__(
            **kwargs)
        self.data_root = data_root
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.mode = mode

        if mode == 'inference':
            self.image_path_list = self.get_inference_path_list()
            self.image_path_list.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
        else:
            self.image_path_list, self.gt_gender_list, self.gt_category_list, \
            self.gt_sport_list = self.parse_annotation()

    def parse_annotation(self):
        """
        标注文件格式如下：
        id,gender,masterCategory,Sports
        0,Men,Apparel,Casual
        1,Men,Apparel,Casual
        2,Men,Accessories,Casual
        3,Men,Footwear,Sports
        :return:
        """

        annotation_path = os.path.join(self.data_root, 'train.csv')
        with codecs.open(annotation_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        image_path_list = []
        gt_gender_list = []
        gt_category_list = []
        gt_sport_list = []
        image_dir = os.path.join(self.data_root, "{}_new".format(self.mode))
        for line in lines[1:]:  # 去除标题行
            img_name, gender, category, sport = line.strip().split(',')
            img_path = os.path.join(image_dir, '{}.jpg'.format(img_name))
            image_path_list.append(img_path)

            # if gender == '' or category == '' or sport == '':
            #     print(line)

            gender, category, sport = self.to_target_id(gender, category, sport)
            gt_gender_list.append(gender)
            gt_category_list.append(category)
            gt_sport_list.append(sport)

        return image_path_list, gt_gender_list, gt_category_list, gt_sport_list

    @classmethod
    def to_target_id(cls, gender, category, sport):
        gender_np = np.zeros((len(gender_list) + 1,))
        category_np = np.zeros((len(category_list) + 1,))
        sport_np = np.zeros((len(sport_list) + 1,))
        # 出现空默认都为0，最后一位是标志位
        if gender != '':
            gender_np[gender_list.index(gender)] = 1
            gender_np[-1] = 1
        if category != '':
            category_np[category_list.index(category)] = 1
            category_np[-1] = 1
        if sport != '':
            sport_np[sport_list.index(sport)] = 1
            sport_np[-1] = 1

        return gender_np, category_np, sport_np

    def get_inference_path_list(self):
        image_path_list = []
        image_dir = os.path.join(self.data_root, 'test_new')
        for image_name in os.listdir(image_dir):
            im_path = os.path.join(image_dir, image_name)
            image_path_list.append(im_path)

        return image_path_list

    def __getitem__(self, index):
        img_path = self.image_path_list[index]
        img = Image.open(img_path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        if self.mode == 'inference':
            return {'image': img}

        gender = self.gt_gender_list[index]
        category = self.gt_category_list[index]
        sport = self.gt_sport_list[index]
        # label = np.zeros(len(self.alpha)).astype('float32')
        # label[gt] = 1.
        if self.target_transforms:
            gender = self.target_transforms(gender)
            category = self.target_transforms(category)
            sport = self.target_transforms(sport)
        return {'image': img,
                'gender': gender,
                'category': category,
                'sport': sport}

    def __len__(self):
        return len(self.image_path_list)


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        # 使用预训练基模型
        self.cnn = self.feature_extractor()

        # gender分类
        self.fc1 = nn.Linear(self.fc_units, len(gender_list))
        # category分类
        self.fc2 = nn.Linear(self.fc_units, len(category_list))
        # sport分类
        self.fc3 = nn.Linear(self.fc_units, len(sport_list))

    @property
    def fc_units(self):
        return 512

    def forward(self, x):
        """

        :param x: [B,C,H,W]
        :return:
        """
        x = self.cnn(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        gender = self.fc1(x)
        category = self.fc2(x)
        sport = self.fc3(x)
        return gender, category, sport
        # return x

    @classmethod
    def feature_extractor(cls):
        return nn.Identity()


class ResNetModel(BaseModel):
    @property
    def fc_units(self):
        return 512

    @classmethod
    def feature_extractor(cls):
        resnet = models.resnet18(pretrained=True)
        return nn.Sequential(*list(resnet.children())[:-2])


class ShuffleModel(BaseModel):
    @property
    def fc_units(self):
        return 1024

    @classmethod
    def feature_extractor(cls):
        shuffle = models.shufflenet_v2_x0_5(pretrained=True)
        return nn.Sequential(*list(shuffle.children())[:-1])


class DenseNetModel(BaseModel):
    @property
    def fc_units(self):
        return 1024

    @classmethod
    def feature_extractor(cls):
        dense = models.densenet121(pretrained=True, memory_efficient=True)
        return nn.Sequential(*list(dense.features),
                             nn.ReLU(inplace=True))


def get_net(net_name):
    if net_name == 'resnet':
        return ResNetModel()
    if net_name == 'shufflenet':
        return ShuffleModel()
    if net_name == 'densenet':
        return DenseNetModel()


def single_loss(y_true, y_predict):
    """

    :param y_true: [B,num_classes]
    :param y_predict: [B,num_classes]
    :return:
    """
    y_predict = torch.softmax(y_predict, dim=-1)
    y_predict = y_predict + 1e-10

    loss = -torch.sum(torch.log(y_predict) * y_true[:, :-1]) / torch.sum(y_true[:, -1])

    return loss


def calculate_loss(gender_true, category_true, sport_true,
                   gender_predict, category_predict, sport_predict):
    loss1 = single_loss(gender_true, gender_predict)
    loss2 = single_loss(category_true, category_predict)
    loss3 = single_loss(sport_true, sport_predict)
    return (loss1 + loss2 + loss3) / 3.


def main(args):
    torch.backends.cudnn.benchmark = True

    data_set = YXSOcrDataset(args.syn_root,
                             transforms=transforms.Compose([
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

    test_data_set = YXSOcrDataset(args.syn_root,
                                  transforms=transforms.Compose([
                                      transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                  ]),
                                  target_transforms=transforms.Lambda(lambda x: torch.from_numpy(x)),
                                  mode='inference')
    test_data_loader = DataLoader(test_data_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                  num_workers=args.workers)

    # 初始化网络
    net = get_net(args.net)
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
    # 加载预训练模型
    if args.init_epoch > 0:
        checkpoint = torch.load(os.path.join(args.output_dir,
                                             'yxs74.{}.{:03d}.pth'.format(args.net, args.init_epoch)),
                                map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        net.load_state_dict(checkpoint['model'])

    # 训练
    for epoch in range(args.init_epoch, args.epochs):
        epoch_loss = 0

        net.train()
        for sample in tqdm(data_loader):
            image = sample['image'].to(device)
            gender = sample['gender'].to(device)
            category = sample['category'].to(device)
            sport = sample['sport'].to(device)

            gender_predict, category_predict, sport_predict = net(image)  # [B,N,C]
            loss = calculate_loss(gender, category, sport,
                                  gender_predict, category_predict, sport_predict)
            # 梯度更新
            net.zero_grad()
            loss.backward()
            optimizer.step()
            # 当前轮的loss
            epoch_loss += loss.item() * image.size(0)

        # 更新lr
        lr_scheduler.step(epoch)

        epoch_loss = epoch_loss / len(data_loader.dataset)
        # 打印日志,保存权重
        print('Epoch: {}/{} loss: {:03f}'.format(epoch + 1, args.epochs, epoch_loss))

        # 保存模型
        if args.output_dir:
            checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch + 1,
                'args': args}
            torch.save(checkpoint,
                       os.path.join(args.output_dir, 'yxs74.{}.{:03d}.pth'.format(args.net, epoch + 1)))

        # 预测
        if (epoch + 1) % 5 == 0:
            inference(net, test_data_loader, epoch)


def inference(net, data_loader, epoch):
    net.eval()

    gender_id_list = []
    category_id_list = []
    sport_id_list = []
    for sample in tqdm(data_loader):
        image = sample['image'].to(device)
        gender_logits, category_logits, sport_logits = net(image)

        _, gender_ids = torch.max(gender_logits, dim=-1)
        _, category_ids = torch.max(category_logits, dim=-1)
        _, sport_ids = torch.max(sport_logits, dim=-1)

        gender_id_list.append(gender_ids.cpu().detach().numpy())
        category_id_list.append(category_ids.cpu().detach().numpy())
        sport_id_list.append(sport_ids.cpu().detach().numpy())
    gender_id_np = np.concatenate(gender_id_list, 0)
    category_id_np = np.concatenate(category_id_list, 0)
    sport_id_np = np.concatenate(sport_id_list, 0)
    # id to name
    gender_name_np = np.vectorize(lambda i: gender_list[i])(gender_id_np)
    category_name_np = np.vectorize(lambda i: category_list[i])(category_id_np)
    sport_name_np = np.vectorize(lambda i: sport_list[i])(sport_id_np)

    with codecs.open('answer.{:03d}.csv'.format(epoch + 1), mode='w', encoding='utf-8') as writer:
        for im_path, gender, category, sport in zip(data_loader.dataset.image_path_list,
                                                    gender_name_np,
                                                    category_name_np,
                                                    sport_name_np):
            writer.write('{},{},{},{}\n'.format(os.path.splitext(os.path.basename(im_path))[0],
                                                gender,
                                                category,
                                                sport))


def test_model():
    import torchsummary
    net = ShuffleModel()
    print(net.training)
    torchsummary.summary(net, input_size=(3, 224, 224))


# test_model()

if __name__ == '__main__':
    """
    Usage:
    source activate pytorch
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export CUDA_VISIBLE_DEVICES=0
    cd /home/mydir/code
    python demo74.py -i /home/mydir/dataset/yanxishe_74 --device cuda --epochs 193 --init-epoch 193
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--syn-root', type=str, default='/home/mydir/dataset/yanxishe_74')
    parser.add_argument('-n', '--net', type=str, default='resnet', help='resnet|shufflenet|densenet')
    parser.add_argument("--device", type=str, default='cpu', help="cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=80, help="epochs")
    parser.add_argument("--init-epoch", type=int, default=0, help="init epoch")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--step-size", type=int, default=20, help="step size")
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight decay (default: 0)')
    parser.add_argument("--workers", type=int, default=4, help="number of workers")
    parser.add_argument('--output-dir', default='./output', help='path where to save')
    arguments = parser.parse_args(sys.argv[1:])
    device = torch.device(
        'cuda' if arguments.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    main(arguments)
