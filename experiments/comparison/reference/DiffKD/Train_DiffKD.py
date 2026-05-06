import os
import csv
import copy
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from loguru import logger
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from DiffKD import DiffKD
from ResTeacher import resnet18
from UASRData import UASRDataset
from CNNStudent import CNNStudent


parser = argparse.ArgumentParser()
parser.add_argument('--num_train_test', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', default=0.001)
parser.add_argument('--weight_decay', default=0.00005)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

transform = transforms.Compose([
    transforms.Resize(size=(224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def criterion_feature(s, t):
    s = torch.flatten(s, 1)
    t = torch.flatten(t, 1)
    return torch.mean(1.0 - torch.cosine_similarity(s, t))


if __name__ == '__main__':

    snapshot_path = './CheckPoints/DeepShip_DiffKD/'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logger.add(snapshot_path + 'log_{time:YYYY-MM-DD_HH-mm-ss}.log', encoding='utf-8', level='INFO')
    logger.info(str(args))

    for cross in range(args.num_train_test):
        logger.info(f'The {cross + 1:02d} training and testing begins')

        sub_folder = snapshot_path + str(cross + 1) + '/'
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        train_path = './CSVSubFolder/DeepShip/' + str(cross + 1) + '_train.csv'
        train_mel = pd.read_csv(train_path)['mel_list'].tolist()
        train_label = pd.read_csv(train_path)['label_list'].tolist()

        test_path = './CSVSubFolder/DeepShip/' + str(cross + 1) + '_test.csv'
        test_mel = pd.read_csv(test_path)['mel_list'].tolist()
        test_label = pd.read_csv(test_path)['label_list'].tolist()

        train_dataset = UASRDataset(mel_list=train_mel, label_list=train_label, transform=transform)
        test_dataset = UASRDataset(mel_list=test_mel, label_list=test_label, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_batch_num = len(train_loader)
        test_batch_num = len(test_loader)

        teacher = resnet18(pretrained=True).cuda()
        teacher.load_state_dict(torch.load('./DeepShip_Teacher/' + str(cross + 1) + '/best_acc.pth'))
        teacher.eval()
        student = CNNStudent().cuda()
        # student.load_state_dict(torch.load('./CheckPoints/DeepShip_Student/' + str(cross + 1) + '/best_acc.pth'))
        student.train()
        model = DiffKD(student=student, teacher=teacher, student_channels=256, teacher_channels=256).cuda()

        criterion_task = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_model_acc = copy.deepcopy(model.state_dict())
        min_model_loss = copy.deepcopy(model.state_dict())
        best_acc, best_epoch = 0.0, 0.0
        min_loss, min_epoch = np.inf, 0.0
        train_loss_collect, train_acc_collect, test_acc_collect = [], [], []

        f_log_csv = open(sub_folder + 'log.csv', 'w', encoding='utf-8', newline='')
        f_log = csv.writer(f_log_csv)
        f_log.writerow(['epoch', 'train loss', 'train acc', 'test acc'])
        f_log_csv.flush()

        for epoch in range(1, args.num_epochs + 1):
            start_time = time.time()

            model.train()
            train_count = 1
            train_loss, train_acc = 0.0, 0.0

            for index, (mel, label) in enumerate(train_loader):
                logger.info(
                    f'cross: {cross + 1} / {args.num_train_test} '
                    f'epoch: {epoch} / {args.num_epochs} train: {train_count} / {train_batch_num}')
                train_count += 1
                mel, label = mel.cuda(), label.cuda()
                s_out, loss_diffkd = model(mel)
                loss_task = criterion_task(s_out, label)
                loss = loss_task + loss_diffkd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step((epoch - 1) + index / len(train_loader))
                train_loss += loss.item()
                train_acc += (s_out.max(1)[1] == label).float().mean().item()

            train_loss /= train_batch_num
            train_acc /= train_batch_num
            train_loss_collect.append(train_loss)
            train_acc_collect.append(train_acc)

            model.eval()
            test_count = 1
            test_acc = 0.0
            with torch.no_grad():
                for mel, label in test_loader:
                    logger.info(
                        f'cross: {cross + 1} / {args.num_train_test} '
                        f'epoch: {epoch} / {args.num_epochs} test: {test_count} / {test_batch_num}')
                    test_count += 1
                    mel, label = mel.cuda(), label.cuda()
                    s_out, _ = model(mel)
                    test_acc += (s_out.max(1)[1] == label).float().mean().item()

            test_acc /= test_batch_num
            test_acc_collect.append(test_acc)

            end_time = time.time()
            cost_time = end_time - start_time

            if epoch % 1 == 0:
                logger.info(
                    f'cost time: {cost_time} epoch: {epoch} / {args.num_epochs} '
                    f'train loss: {train_loss} train acc: {train_acc} test acc: {test_acc}')

            f_log.writerow([epoch, train_loss, train_acc, test_acc])
            f_log_csv.flush()

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                best_model_acc = copy.deepcopy(model.state_dict())

            if train_loss < min_loss:
                min_loss = train_loss
                min_epoch = epoch
                min_model_loss = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model_acc)
        torch.save(model.state_dict(), sub_folder + '/best_acc.pth')

        model.load_state_dict(min_model_loss)
        torch.save(model.state_dict(), sub_folder + '/min_loss.pth')

        final_model = copy.deepcopy(model.state_dict())
        model.load_state_dict(final_model)
        torch.save(model.state_dict(), sub_folder + '/final.pth')

        logger.info(f'best acc: {best_acc}, best epoch: {best_epoch}')
        logger.info(f'min loss: {min_loss}, min epoch: {min_epoch}')
        logger.info(f'The {cross + 1:02d} training and test is over')
        f_log_csv.close()

        fig = plt.figure(figsize=(24, 8))
        fig_1 = plt.subplot(1, 2, 1)
        plt.plot(train_loss_collect)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train loss'])
        plt.title('loss curve')

        fig_2 = plt.subplot(1, 2, 2)
        plt.plot(train_acc_collect)
        plt.plot(test_acc_collect)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train acc', 'test acc'])
        plt.title('acc curve')
        plt.savefig(sub_folder + './curve.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

    logger.info(f'{args.num_train_test} training and test is over')
