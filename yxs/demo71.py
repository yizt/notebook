# -*- coding: utf-8 -*-
"""
 @File    : demo.py
 @Time    : 2020/7/9 上午7:55
 @Author  : yizuotian
 @Description    :
"""
import argparse
import codecs
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm

trans = transforms.Compose([
    transforms.Resize((128, 32)),  # [h,w]
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def get_random_point(h, w, edge):
    """
    获取指定边上随机的点
    :param h:
    :param w:
    :param edge:
    :return (x,y):
    """
    if edge == 't':
        return np.random.randint(w), 0
    elif edge == 'b':
        return np.random.randint(w), h
    elif edge == 'l':
        return 0, np.random.randint(h)
    else:
        return w, np.random.randint(h)


def random_lines(img_gray, max_lines=2):
    num = np.random.randint(max_lines + 1)
    h, w = img_gray.shape[:2]
    for _ in range(num):
        # 选择4个边上选择两个点
        edges = np.random.choice(['t', 'b', 'r', 'l'], 2, replace=False)
        p1 = get_random_point(h, w, edges[0])
        p2 = get_random_point(h, w, edges[1])
        color = np.random.randint(255)
        cv2.line(img_gray, p1, p2, color, np.random.randint(1, 3))  # 干扰线
    return img_gray


class YXSOcrDataset(Dataset):
    def __init__(self, data_root, transforms=None,
                 target_transforms=None, mode='train', **kwargs):
        super(YXSOcrDataset, self).__init__(
            **kwargs)
        self.data_root = data_root
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.mode = mode

        self.image_path_list, self.gt_list = self.parse_annotation()
        self.alpha = self.init_alpha()

        if mode == 'inference':
            self.image_path_list = self.get_inference_path_list()
            self.image_path_list.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
            self.image_list = self.pre_load()
        else:
            self.image_list, self.target_list = self.pre_load()

    def parse_annotation(self):
        """
        标注文件格式如下：
        filename,label
        0.jpg,去岸没谨峰福
        1.jpg,蜕缩蝠缎掐助崔
        2.jpg,木林焙袒舰酝凶厚
        :return:
        """
        annotation_path = os.path.join(self.data_root, 'train.csv')
        with codecs.open(annotation_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        image_path_list = []
        gt_list = []
        image_dir = os.path.join(self.data_root, self.mode)
        for line in lines[1:]:  # 去除标题行
            img_name, text = line.strip().split(',')
            img_path = os.path.join(image_dir, img_name)
            image_path_list.append(img_path)
            gt_list.append(text)
        return image_path_list, gt_list

    def get_inference_path_list(self):
        image_path_list = []
        image_dir = os.path.join(self.data_root, 'test')
        for image_name in os.listdir(image_dir):
            im_path = os.path.join(image_dir, image_name)
            image_path_list.append(im_path)
        return image_path_list

    def init_alpha(self):
        alpha = set()
        for text in self.gt_list:
            alpha = alpha.union(set(text))
        alpha = list(alpha)
        alpha.sort()
        alpha = ''.join(alpha)
        return alpha

    def pre_load(self):
        im_list = []
        target_list = []
        if self.mode == 'train':
            for im_path, text in zip(self.image_path_list, self.gt_list):
                im = cv2.imread(im_path, 0)
                for i, char in enumerate(text):
                    im_list.append(im[:, i * 25:(i + 1) * 25])
                    target_list.append(self.alpha.index(char))
                    # assert im.shape[1] // 25 == len(text)
            print('preload done!')
            return im_list, target_list
        else:
            for im_path in self.image_path_list:
                im = cv2.imread(im_path, 0)
                w = im.shape[1]
                for i in range(w // 25):
                    im_list.append(im[:, i * 25:(i + 1) * 25])
            print('preload done!')
            return im_list

    def __getitem__(self, index):
        img = self.image_list[index]
        if self.mode == 'train':
            img = random_lines(img.copy())
        img = Image.fromarray(img)
        if self.transforms:
            img = self.transforms(img)

        if self.mode == 'inference':
            return {'image': img}

        gt = self.target_list[index]
        # label = np.zeros(len(self.alpha)).astype('float32')
        # label[gt] = 1.
        if self.target_transforms:
            label = self.target_transforms(np.array(gt))
        return {'image': img,
                'target': label}

    def __len__(self):
        return len(self.image_list)


class ChannelPool(nn.Module):
    def forward(self, x):
        """

        :param x: [B,C,H,W]
        :return:  [B,2,H,W]
        """
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """

        :param x: [B,C,H,W]
        :return: [B,C,W]
        """
        x_compress = self.compress(x)  # [B,2,H,W]
        x_out = self.spatial(x_compress)  # [B,1,H,W]
        scale = self.sigmoid(x_out)  # broadcasting
        x = x * scale
        x = torch.sum(x, dim=2)  # [B,C,H,W]=>[B,C,W]
        return x


class BaseModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self.num_classes = num_classes
        # 第一层是单通道
        self.conv = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(64)
        # 使用预训练基模型
        self.cnn = self.feature_extractor()

        self.spatial_gate = SpatialGate()
        # 分类
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """

        :param x: [B,C,H,W]
        :return:
        """
        x = F.relu(self.bn(self.conv(x)), True)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        x = self.cnn(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        x = self.spatial_gate(x)  # [B,C,H,W]=>[B,C,1]

        x = x.squeeze()  # [B,C,W]=>[B,C,1]
        x = self.fc(x)
        return x

    @classmethod
    def feature_extractor(cls):
        return nn.Identity()


class ResNetModel(BaseModel):
    @classmethod
    def feature_extractor(cls):
        resnet = models.resnet18(pretrained=True)
        return nn.Sequential(*list(resnet.children())[4:-2])


def train(args):
    torch.backends.cudnn.benchmark = True

    data_set = YXSOcrDataset(args.syn_root,
                             transforms=trans,
                             target_transforms=transforms.Lambda(lambda x: torch.from_numpy(x)))
    train_sampler = torch.utils.data.RandomSampler(data_set)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, sampler=train_sampler,
                             num_workers=args.workers)

    #
    net = ResNetModel(len(data_set.alpha))
    net.train()
    net.to(device)

    optimizer = optim.Adadelta(net.parameters(), weight_decay=args.weight_decay)

    # 加载预训练模型
    if args.init_epoch > 0:
        checkpoint = torch.load(os.path.join(args.output_dir,
                                             'yxs.{:03d}.pth'.format(args.init_epoch)),
                                map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        net.load_state_dict(checkpoint['model'])

    # 训练
    for epoch in range(args.init_epoch, args.epochs):
        epoch_loss = 0

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

        epoch_loss = epoch_loss / len(data_loader.dataset)
        # 打印日志,保存权重
        print('Epoch: {}/{} loss: {:03f}'.format(epoch + 1, args.epochs, epoch_loss))

        # 保存模型
        if args.output_dir:
            checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch + 1,
                'args': args}
            torch.save(checkpoint,
                       os.path.join(args.output_dir, 'yxs.{:03d}.pth'.format(epoch + 1)))

    return net


def inference(args, net):
    net.eval()
    data_set = YXSOcrDataset(args.syn_root,
                             transforms=trans,
                             target_transforms=transforms.Lambda(lambda x: torch.from_numpy(x)),
                             mode='inference')
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.workers)
    class_id_list = []
    for sample in tqdm(data_loader):
        image = sample['image'].to(device)
        outputs = net(image)  # [B,N,C]
        _, class_ids = torch.max(outputs, dim=-1)
        class_id_list.append(class_ids.cpu().detach().numpy())

    class_id_np = np.concatenate(class_id_list, axis=0)
    class_name_np = np.vectorize(lambda i: data_set.alpha[i])(class_id_np)
    with codecs.open('answer.{:03d}.csv'.format(args.epochs), mode='w', encoding='utf-8') as writer:
        idx = 0
        for im_path in data_set.image_path_list:
            im = cv2.imread(im_path, 0)
            h, w = im.shape[:2]
            shift = w // 25

            text = class_name_np[idx:idx + shift]
            writer.write('{},{}\n'.format(os.path.splitext(os.path.basename(im_path))[0], ''.join(text)))
            idx += shift


def test_model():
    import torchsummary
    net = ResNetModel(100)
    print(net.training)
    torchsummary.summary(net, input_size=(1, 128, 32))


if __name__ == '__main__':
    """
    Usage:
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export CUDA_VISIBLE_DEVICES=0
    python demo2.py -i /home/mydir/dataset/71_OCR_2 --device cuda --epochs 193 --init-epoch 193
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--syn-root', type=str, default='/Users/admin/Downloads/71_OCR_2')
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
