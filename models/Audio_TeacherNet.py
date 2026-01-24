# Audio TeacherNet
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class Audio_TeacherNet(torch.nn.Module):
    def __init__(self,num_classes=5):
        super(Audio_TeacherNet, self).__init__()
        # https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1696
        # self.processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-large-robust-12-ft-emotion-msp-dim")
        self.processor = Wav2Vec2Processor.from_pretrained(r'/media/hdd1/fubohan/Code/UATR/models/wav2vec2-large-robust-12-ft-emotion-msp-dim',
                                                           local_files_only=True)
        
        # Feature Extractor + Hidden Layers
        # self.wav2Vec = Wav2Vec2Model.from_pretrained("./wav2vec2-large-robust-12-ft-emotion-msp-dim")
        self.wav2Vec = Wav2Vec2Model.from_pretrained(r'/media/hdd1/fubohan/Code/UATR/models/wav2vec2-large-robust-12-ft-emotion-msp-dim',
                                                     local_files_only=True)
        
        # Classification Head
        self.duration_linear = nn.Sequential(
            nn.Linear(1024,128),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Linear(128, num_classes))
        
    def freeze_feature_extractor(self):
        self.wav2Vec.feature_extractor._freeze_parameters()
    
    def process(self, x):
        return self.processor(x, sampling_rate=16000, return_tensors="pt").input_values

    def forward(self, x, output_all_hidden_states=True):
        x=torch.squeeze(x,dim=1)
        # last_hidden_state
        # extract_features
        # hidden_states
        # attentions
        wav2vec_result=self.wav2Vec(x, output_hidden_states=output_all_hidden_states)
        
        # 32, 149, 512
        x_extract_features=wav2vec_result.extract_features
        # 32, 149, 512
        output_extract_features=x_extract_features
        
        # 32, 149, 1024
        x_last_hidden_state=wav2vec_result.last_hidden_state
        # 32, 149, 1024
        output_last_hidden_state=x_last_hidden_state
        
        # 获取所有隐藏状态 [num_layers + 1, B, seq_len, hidden_size]
        all_hidden_states = None
        if output_all_hidden_states and hasattr(wav2vec_result, 'hidden_states'):
            all_hidden_states = wav2vec_result.hidden_states  # tuple of [B, seq_len, hidden_size]
            # 将所有隐藏状态堆叠成 [num_layers + 1, B, seq_len, hidden_size]
            if all_hidden_states:
                all_hidden_states = torch.stack(all_hidden_states, dim=0)
        
        # 32, 149, 128
        x=torch.mean(x_last_hidden_state,dim=1)
        x=self.duration_linear(x)
        
        return x, output_extract_features, output_last_hidden_state, all_hidden_states

def build_Audio_TeacherNet(num_classes=5, checkpoint_path=None):
    model=Audio_TeacherNet(num_classes=num_classes)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"Loaded Audio_TeacherNet weights from {checkpoint_path}")
    return model

if __name__ == "__main__":
    model=build_Audio_TeacherNet(num_classes=4, checkpoint_path='/media/hdd1/fubohan/Code/UATR/models/Audio_Teacher_DeepShip_622/checkpoints/Student.pth')
    print(f"参数总量: {sum(p.numel() for p in model.parameters())}")
    input=torch.randn(4,1,48000)
    output,feat_ext,feat_last,all_hidden_states=model(input)
    print(output.shape)
    print(feat_ext.shape)
    print(feat_last.shape)
    if all_hidden_states is not None:
        print(f"所有隐藏状态形状: {all_hidden_states.shape}")  # [num_layers + 1, B, seq_len, hidden_size]
