import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from transformers import AutoTokenizer
from a02_my_model import MyModel

class MyDataset(Dataset):
    def __init__(self, split: str):
        super(MyDataset, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MyModel.model_name)
        self.dataset = load_from_disk("04_demo_4/data/ChnSentiCorp")[split]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def collate_func(self, data):
        # print()
        # print(data)
        input = [i["text"] for i in data]
        target = [i["label"] for i in data]

        input_coded = self.tokenizer(
            input, 
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        target = torch.LongTensor(target)
        return (input_coded, target)

    
if __name__ == "__main__":
    my_dataset = MyDataset("train")
    print(len(my_dataset))
    print(my_dataset[0])