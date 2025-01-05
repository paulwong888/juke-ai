import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from a00_constant import bert_base_chinese_model_path

class MyDataset(Dataset):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = bert_base_chinese_model_path

    def __init__(self, data_dir, split):
        # self.my_data_set = load_dataset(path="csv", data_dir=weibo_file_dir, split=split)
        self.my_data_set = load_dataset(path="csv", data_dir=data_dir, split=split)
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(MyDataset.model_name)

    def __len__(self):
        return len(self.my_data_set)

    def __getitem__(self, index: int):
        return self.my_data_set[index]
    
    def collate_fun(self, data: list):
        input = [item["text"] for item in data]
        target = [item["label"] for item in data]

        input_coded = self.tokenizer.batch_encode_plus(
            input, max_length=512, truncation=True,
            padding="max_length", return_tensors="pt"
        )

        for key in input_coded.keys():
            input_coded[key] = input_coded[key].to(MyDataset.device)
        
        target = torch.LongTensor(target).to(MyDataset.device)

        return (input_coded, target)
    
    def build_data_loader(self, batch_size: int):
        return DataLoader(
            self, batch_size=batch_size,
            shuffle=True, drop_last=True,
            collate_fn=lambda item: self.collate_fun(item)
        )
    
if __name__ == "__main__":
    from a00_constant import (
        weibo_file_dir, 
        weibo_over_sample_file_dir,
        weibo_under_sample_file_dir,
        weibo_over_sample_train_file_path
    )
    my_dataset = MyDataset(weibo_under_sample_file_dir, "validation")
    print(my_dataset[1])
    print(len(my_dataset))
    data_list = my_dataset.my_data_set.select(range(5))
    [print(item) for item in data_list]
    print(my_dataset.collate_fun(data_list))