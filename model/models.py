import torch
import torch.nn as nn

import model.blocks as blocks
import torch.nn.functional as F

class TransModel(nn.Module):
    def __init__(self, num_genomic_features, mid_hidden = 512, record_attn = False):
        super(TransModel, self).__init__()
        print('Initializing TransModel')
        self.conv1 = nn.Sequential(
            nn.Conv1d(6, 512, 129, 128, 64),
            #nn.Conv1d(1, 512, 129, 128, 64),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.attn = blocks.AttnModule(hidden = mid_hidden, record_attn = record_attn, inpu_dim =512)
        self.conv2 = nn.Conv1d(512, 1, 3, 1, 1)
        self.Linear4 = nn.Linear(in_features =512, out_features = 1024)
        self.record_attn = record_attn
        self.dropout = nn.Dropout(p=0.1)
        
    def move_feature_forward(self, x):
        '''
        Input dim:
        batch_size, length, feature
        to: 
        batch_size, feature, length
        '''
        return x.transpose(1, 2).contiguous() 
        
    def forward(self, x):
        '''
        Input feature:
        batch_size, length = 65536, feature_dim =6
        '''
        x = self.move_feature_forward(x).float()
        #seq = x[:, :5, :]
        #epi = x[:, 5:, :]
        #x = seq
        #x = epi
        
        #seq = torch.zeros_like(x[:, :5, :])
        #x[:, :5, :] = seq
        
        x = self.conv1(x)
        x = self.move_feature_forward(x).float()
        if self.record_attn:
            x, attn_weights = self.attn(x)
        else:
            x = self.attn(x)
        x = self.dropout(x)
        x = self.move_feature_forward(x).float()
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.Linear4(x).squeeze(1)
        x = F.relu(x)
        if self.record_attn:
            return x, attn_weights
        else:
            return x         
        
if __name__ == '__main__':
    main()
