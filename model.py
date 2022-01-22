import torch.nn as nn
import torch.nn.functional as F

class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 2048)

        self.v_stream = nn.Linear(2048, 512)
        self.v = nn.Linear(512, 1)
        self.a_stream = nn.Linear(2048, 512)
        self.a = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        
        v = F.relu(self.v_stream(x))
        v = self.v(v)
        a = F.relu(self.a_stream(x))
        a = self.a(a)

        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
