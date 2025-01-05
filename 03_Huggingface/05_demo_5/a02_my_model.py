import torch
from torch import nn
from transformers import BertModel
from a01_my_dataset import MyDataset

class MyModel(nn.Module):

    model = BertModel.from_pretrained(MyDataset.model_name).to(MyDataset.device)

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 2)

    def forward(self, X):
        with torch.no_grad():
            X = self.model(**X)
        # return self.fc(X.last_hidden_state[:,0])
        return self.fc(X.pooler_output)
    
if __name__ == "__main__":
    my_dataset = MyDataset("test")
    for item, _ in my_dataset.build_data_loader(10):
        input = item
        break
    my_model = MyModel().to(MyDataset.device)
    my_model.load_state_dict(torch.load("03_Huggingface/05_demo_5/params/1_bert.pth", map_location=MyDataset.device))
    print(my_model.forward(input))