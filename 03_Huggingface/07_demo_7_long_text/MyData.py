from torch.utils.data import Dataset
from datasets import load_dataset
from a00_constant import news_file_dir

class MyDataset(Dataset):
    def __init__(self,split):
        #从磁盘加载数据
        self.dataset = load_dataset(path="csv",data_dir=news_file_dir, split="train")
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text,label

if __name__ == '__main__':
    dataset = MyDataset("test")
    for data in dataset:
        print(data)