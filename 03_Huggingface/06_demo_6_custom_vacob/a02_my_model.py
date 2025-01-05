import torch
from torch import nn
from a01_my_data_set import MyDataset
from transformers import BertModel

class MyModel(nn.Module):

    model = BertModel.from_pretrained(MyDataset.model_name).to(MyDataset.device)

    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(768, 8)
    
    def forward(self, X):
        with torch.no_grad():
            X = MyModel.model.forward(**X)
        return self.linear.forward(X.pooler_output)

    def get_instance(self):
        return self.to(MyDataset.device)

if __name__ == "__main__":
    my_dataset = MyDataset("test")
    for item, _ in my_dataset.build_data_loader(10):
        input = item
        break
    
    my_model = MyModel().get_instance()
    with torch.no_grad():
        output = my_model.forward(input)
    print(output)
    