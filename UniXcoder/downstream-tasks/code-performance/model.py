# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.scorer = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),     
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),     
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, code_inputs=None):
        outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
        outputs = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
        scores = self.scorer(outputs)
        return scores.squeeze(-1).float()
    