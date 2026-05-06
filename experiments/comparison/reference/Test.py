import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import roc_auc_score, f1_score

from MMKD import MMKD
from ResTeacher import resnet18
from UASRData import UASRDataset
from CNNStudent import CNNStudent


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
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


if __name__ == '__main__':

    teacher = resnet18(pretrained=False).cuda()
    student = CNNStudent().cuda()
    model = MMKD(teacher=teacher, student=student, mask_ratio=0.1, depth=4).cuda()
    model.load_state_dict(torch.load('./CheckPoints/ShipsEar_MMKD/1/best_acc.pth'))
    model.eval()

    test_path = './CSVSubFolder/ShipsEar/1_test.csv'
    test_mel = pd.read_csv(test_path)['mel_list'].tolist()
    test_label = pd.read_csv(test_path)['label_list'].tolist()

    test_dataset = UASRDataset(mel_list=test_mel, label_list=test_label, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred, y_logit = [], [], []
    test_acc = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output, _, _, _, _, _, _, _ = model(data)
            test_acc += (output.max(1)[1] == target).float().mean().item()
            pred = torch.argmax(output, 1)
            logit = F.softmax(output, dim=1)
            y_true.extend(target)
            y_pred.extend(pred)
            y_logit.append(logit.cpu())

    y_true = [i.item() for i in y_true]
    y_pred = [i.item() for i in y_pred]
    y_score = torch.cat(y_logit, dim=0)
    y_score = y_score.detach().numpy()
    acc = test_acc / len(test_loader)
    f1 = f1_score(y_true, y_pred, average='micro')
    auc = roc_auc_score(y_true, y_score, multi_class='ovo')
    print(acc)
    print(f1)
    print(auc)
