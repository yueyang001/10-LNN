import os
import sys
import copy
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import albumentations as al
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from ResNet_FSP import resnet18
from Student_FSP import StudentNet
from ReDataset import ReDataset
from Distillation_FSP import DistillationModel


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', default=0.001)
parser.add_argument('--weight_decay', default=0.00005)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))

cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

model_T = resnet18(pretrained=True).cuda()
model_T.load_state_dict(torch.load('./Checkpoints_EarShip_Teacher/best_acc.pth'))
model_S = StudentNet().cuda()
model = DistillationModel(model_T, model_S).cuda()

train_file_path = './path/train_EarShip.csv'
test_file_path = './path/test_EarShip.csv'

transform = al.Resize(224, 224, p=1)
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.5], std=[0.5])


if __name__ == '__main__':

    snapshot_path = './Checkpoints_EarShip_FSP/'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    TrainDataset = ReDataset(file_path=train_file_path, transform=transform, to_tensor=to_tensor, normalize=normalize)
    TrainLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)

    TestDataset = ReDataset(file_path=test_file_path, transform=transform, to_tensor=to_tensor, normalize=normalize)
    TestLoader = DataLoader(TestDataset, batch_size=batch_size, shuffle=False)

    dataset = {'train': TrainLoader, 'test': TestLoader}
    dataset_sizes = {'train': len(TrainLoader), 'test': len(TestLoader)}
    dataset_length = {'train': len(TrainDataset), 'test': len(TestDataset)}

    criterion_task = nn.CrossEntropyLoss()
    criterion_KL = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    writer = SummaryWriter(snapshot_path + 'log')

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0
    best_epoch = 0
    num_epochs = args.num_epochs
    train_loss_graph = []
    test_loss_graph = []
    train_acc_graph = []
    test_acc_graph = []
    T = 3

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0
            running_acc = 0

            for i, (inputs, label) in enumerate(dataset[phase]):
                inputs, label = inputs.float().cuda(), label.cuda()

                with torch.set_grad_enabled(phase == 'train'):
                    teacher_output, student_output, loss1, loss2 = model(inputs)
                    pred = torch.argmax(student_output, 1)
                    loss_task = criterion_task(student_output, label)
                    student_output_T = F.log_softmax(student_output / T, dim=1)
                    teacher_output_T = F.softmax(teacher_output / T, dim=1)
                    loss_kl = criterion_KL(student_output_T, teacher_output_T)
                    loss = loss_task + loss_kl + loss1 + loss2

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step(epoch + i / len(TrainLoader))

                running_loss += loss.item()
                running_acc += torch.sum(pred == label)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc / dataset_length[phase] * 100

            logging.info(phase + '_epoch %d : loss : %f  acc : %f' % (epoch + 1, epoch_loss, epoch_acc))

            writer.add_scalar(phase + '_loss/loss', epoch_loss, epoch + 1)
            writer.add_scalar(phase + '_acc', epoch_acc, epoch + 1)

            if phase == 'train':
                train_loss_graph.append(epoch_loss)
                train_acc_graph.append(epoch_acc)
            else:
                test_loss_graph.append(epoch_loss)
                test_acc_graph.append(epoch_acc)

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch + 1
                best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), snapshot_path + '/best_acc.pth')

    final_model = copy.deepcopy(model.state_dict())
    model.load_state_dict(final_model)
    torch.save(model.state_dict(), snapshot_path + '/final_epoch.pth')

    logging.info('Best_acc : %f' % best_acc)
    logging.info('Best_acc_epoch : %f' % best_epoch)

    fig = plt.figure(figsize=(24, 8))
    fig_1 = plt.subplot(1, 2, 1)
    plt.plot(train_loss_graph)
    plt.plot(test_loss_graph)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train loss', 'Test loss'])
    plt.title('Loss curve')

    fig_2 = plt.subplot(1, 2, 2)
    plt.plot(train_acc_graph)
    plt.plot(test_acc_graph)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train acc', 'Test acc'])
    plt.title('Acc curve')
    plt.savefig(snapshot_path + './show_result.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
