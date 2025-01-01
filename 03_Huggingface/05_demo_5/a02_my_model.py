import torch
from torch import nn
from transformers import BertModel

class MyModel(nn.Module):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = r"03_Huggingface/04_demo_4/trasnFormers_test/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
    model = BertModel.from_pretrained(model_name).to(device)

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 2)

    def forward(self, X):
        with torch.no_grad():
            X = self.model(**X)
        # return self.fc(X.last_hidden_state[:,0])
        return self.fc(X.pooler_output)
    
if __name__ == "__main__":
    my_model = MyModel().to(MyModel.device)
    my_model.load_state_dict(torch.load("03_Huggingface/05_demo_5/params/1_bert.pth", map_location=MyModel.device))