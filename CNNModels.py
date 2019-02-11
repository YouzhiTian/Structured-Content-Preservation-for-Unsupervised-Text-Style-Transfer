import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnTextClassifier(nn.Module):
    def __init__(self, vocab_size,emb_size,num_filters,num_classes, window_sizes=(3, 4, 5)):
        super(CnnTextClassifier, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, [window_size, emb_size], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.fc = nn.Linear(num_filters * len(window_sizes), num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)       
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        
            x2 = torch.squeeze(x2, -1)  
            x2 = F.max_pool1d(x2, x2.size(2))  
            xs.append(x2)
        x = torch.cat(xs, 2)            
        x = x.view(x.size(0), -1)       
        logits = self.fc(x)             
        
        #True accuracy
        probs = F.softmax(logits)       
        classes = torch.max(probs, 1)[1]

        return probs, classes
