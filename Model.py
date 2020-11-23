import torch.nn as nn

class Project(nn.Module):
    def __init__(self):
        super(Project,self).__init__()
        self.conv = nn.Conv1d(1,2,3,padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(2,4,2,padding=1)
        self.linear1 = nn.Linear(3*4,8)
        self.linear2 = nn.Linear(8,4)
        self.linear3 = nn.Linear(4,2)
        
    def forward(self, input):
        x = self.conv(input.view(len(input),1,10))
        x = self.pool(x)
        x = self.conv1(x)
        x = x.view(-1,3*4)
        x = self.linear3(self.linear2(self.linear1(x)))
        return x