import torch.nn as nn
import sys
sys.path.append('../utils/')
from FilterPreprocess import FilterPreprocess
from FuzzyViT import FuzzyViT
from FuzzyDecisionLayer import FuzzyDecisionLayer

class FuzzyAudioModel(nn.Module):
    def __init__(self, sample_rate=44100, num_classes=1000, dropout_prob=0.1, use_fuzzy_dropout=True):
        super(FuzzyAudioModel, self).__init__()
        
        self.processor = FilterPreprocess(sample_rate=sample_rate)
        self.model_vit = FuzzyViT(num_classes=num_classes, dropout_prob=dropout_prob, use_fuzzy_dropout=use_fuzzy_dropout)
        
        self.model_decision = None 

    def forward(self, waveform, sample_rate):
        mel1, mel2, mel3 = self.processor.process_audio(waveform, sample_rate)
        
        output_vit = self.model_vit(mel1, mel2, mel3)
        
        if self.model_decision is None:
            input_dim = output_vit.shape[1]
            self.model_decision = FuzzyDecisionLayer(input_dim).to(output_vit.device)
        
        output_decision = self.model_decision(output_vit)
        
        return output_decision
