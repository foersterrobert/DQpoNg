import torch.nn as nn
import torch.nn.functional as F
import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1024)
        self.linear4 = nn.Linear(1024, 256)
        self.linear5 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x

    def act(self, state):
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
            q_values = self(state_t.unsqueeze(0))
            max_q_index = torch.argmax(q_values, dim=1)[0]
            action = max_q_index.detach().item()
        return action

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
