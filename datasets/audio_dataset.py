import warnings
warnings.filterwarnings('ignore')


import os
import sys
import glob
from PIL import Image
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor


#图像预处理
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
validation_test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])



# data_flag: train / validation / test
# data_type: wav_s@wav_t@mel@cqt 
#            wav_s / wav_t / mel / cqt
# transform: train_transform / validation_test_transform
class AudioDataset(Dataset):
    def __init__(self,data_dir,data_flag,data_type,transform):
        super(AudioDataset,self).__init__()
        self.sr=16000
        self.transform=transform
        self.dataset_dir=os.path.join(data_dir,data_flag)
        
        self.data_type=data_type.split('@')
        
        # wav
        self.wav_dataset_dir=os.path.join(self.dataset_dir,'wav')
        # mel
        self.mel_dataset_dir=os.path.join(self.dataset_dir,'mel')
        # cqt
        self.cqt_dataset_dir=os.path.join(self.dataset_dir,'cqt')
        
        # category names
        self.category_names=os.listdir(self.wav_dataset_dir)
        # classes
        self.classes=self.category_names
        self.classes=self.category_names
        if len(self.category_names)==5:
            self.category_names = ['A', 'D', 'C', 'B', 'E']
        elif len(self.category_names)==4:
            self.category_names = ['passengership', 'tanker', 'cargo', 'tug']
        
        # All sample_paths
        #   single wav_path
        #   single mel_path
        #   single cqt_path
        #   single sample index
        self.sample_paths=[]
        for category_index,category_name in enumerate(self.category_names,0):
            # wav
            category_wav_dataset_dir=os.path.join(self.wav_dataset_dir,category_name)
            # mel
            category_mel_dataset_dir=os.path.join(self.mel_dataset_dir,category_name)
            # cqt
            category_cqt_dataset_dir=os.path.join(self.cqt_dataset_dir,category_name)
            
            for category_wav_path in glob.glob(os.path.join(category_wav_dataset_dir,'*.wav')):
                # wav
                category_wav_name=os.path.basename(category_wav_path).split('.')[0]
                # mel
                category_mel_name=category_wav_name+'@mel.png'
                category_mel_path=os.path.join(category_mel_dataset_dir,category_mel_name)
                # cqt
                category_cqt_name=category_wav_name+'@cqt.png'
                category_cqt_path=os.path.join(category_cqt_dataset_dir,category_cqt_name)
                
                single_sample_path=[category_wav_path,
                                    category_mel_path,
                                    category_cqt_path,
                                    category_index]
                
                self.sample_paths.append(single_sample_path)
                
        # Online
        # self.processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-large-robust-12-ft-emotion-msp-dim")
        # Offline
        self.processor = Wav2Vec2Processor.from_pretrained(r'/media/hdd1/fubohan/Code/UATR/models/wav2vec2-large-robust-12-ft-emotion-msp-dim',local_files_only=True)
                
    def __getitem__(self, item):
        wav_data_path, \
        mel_data_path, \
        cqt_data_path, \
        label=self.sample_paths[item]
        
        input_data=[]
        
        if 'wav_s' in self.data_type:
            wav_data,sr=torchaudio.load(wav_data_path)
            resampler = torchaudio.transforms.Resample(sr,self.sr)
            wav_data_student = resampler(wav_data)
            input_data.append(wav_data_student)
            
        if 'wav_t' in self.data_type:
            wav_data,sr=torchaudio.load(wav_data_path)
            resampler = torchaudio.transforms.Resample(sr,self.sr)
            wav_data = resampler(wav_data).squeeze().numpy()
            wav_data_teacher=self.processor(wav_data,sampling_rate=self.sr,return_tensors="pt").input_values
            input_data.append(wav_data_teacher)
            
        if 'mel' in self.data_type:
            mel_data=Image.open(mel_data_path).convert('RGB')
            mel_data=self.transform(mel_data)
            input_data.append(mel_data)
            
        if 'cqt' in self.data_type:
            cqt_data=Image.open(cqt_data_path).convert('RGB')
            cqt_data=self.transform(cqt_data)
            input_data.append(cqt_data)


        label = torch.from_numpy(np.array(label))
        label = label.type(torch.int64)

        return input_data,label

    def __len__(self):
        return len(self.sample_paths)



