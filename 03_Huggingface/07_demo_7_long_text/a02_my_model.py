import torch
from a00_constant import bert_model_path, max_length
from a01_my_data_set import device
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class MyModel(nn.Module):
    

    def __init__(self):
        super(MyModel, self).__init__()
        bert_config = BertConfig.from_pretrained(bert_model_path)
        bert_config.max_position_embeddings = max_length
        self.model = BertModel._from_config(bert_config).to(device)
        self.linear = nn.Linear(768, 10)

    def forward(self, X):
        # 由于改了max_length, 必须全参数训练
        # with torch.no_grad(): 
        X = self.model.forward(**X)
        X = X.pooler_output
        # X = nn.Dropout(X)
        return F.dropout(self.linear.forward(X))
    
    def instance(self):
        return self.to(device)
    

if __name__ == "__main__":
    my_model = MyModel()
    print(my_model.model)