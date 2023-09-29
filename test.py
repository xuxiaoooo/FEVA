from utils.FilterPreprocess import FilterPreprocess
from utils.FuzzyViT import FuzzyViT
from utils.FuzzyDecisionLayer import FuzzyDecisionLayer
from utils.FuzzyAudioModel import FuzzyAudioModel
import torchaudio, torch

# processor = FilterPreprocess(sample_rate=44100)
# waveform, sr = torchaudio.load("/home/user/xuxiao/FEVA/datasets/ANRAC/6962/audio.wav")
# mel1, mel2, mel3 = processor.process_audio(waveform, sample_rate=sr)
# # print(mel1.shape, mel2.shape, mel3.shape)
# model_vit = FuzzyViT(num_classes=1000, dropout_prob=0.1, use_fuzzy_dropout=True)
# output_vit = model_vit(mel1, mel2, mel3)
# input_dim = output_vit.shape[1]
# model_decision = FuzzyDecisionLayer(input_dim)
# output_decision = model_decision(output_vit)
# print(output_decision)

# 1. 实例化模型
model = FuzzyAudioModel(sample_rate=44100, num_classes=1000, dropout_prob=0.1, use_fuzzy_dropout=True)

# 如果您有一个预训练的权重文件，您还可以这样加载它：
# model.load_state_dict(torch.load("path_to_pretrained_weights.pth"))
# model.eval()  # 将模型设置为评估模式

# 2. 加载音频数据
waveform, sample_rate = torchaudio.load("/home/user/xuxiao/FEVA/datasets/ANRAC/6962/audio.wav")

# 3. 使用模型进行预测
with torch.no_grad():  # 确保不进行梯度计算
    predictions = model(waveform, sample_rate)

print(predictions)