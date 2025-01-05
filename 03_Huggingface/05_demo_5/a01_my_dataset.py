import torch
from a00_constant import bert_base_chinese_model_path
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer

class MyDataset(Dataset):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = bert_base_chinese_model_path

    def __init__(self, split: str):
        super(MyDataset, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MyDataset.model_name)
        self.dataset = load_from_disk("03_Huggingface/data/ChnSentiCorp")[split]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def collate_func(self, data):
        # print()
        # print(type(data))
        input = [i["text"] for i in data]
        target = [i["label"] for i in data]

        input_coded = self.tokenizer(
            input, 
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        for key in input_coded.keys():
            input_coded[key] = input_coded[key].to(MyDataset.device)
        
        target = torch.LongTensor(target).to(MyDataset.device)
        return (input_coded, target)
    
    

    def build_data_loader(self, batch_size: int):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=True,
            #舍弃最后一个批次的数据，防止形状出错
            drop_last=True,
            collate_fn=lambda x: self.collate_func(x)
        )

    
if __name__ == "__main__":
    my_dataset = MyDataset("test")
    print(len(my_dataset))
    dataset_list = my_dataset.dataset.select(range(5))
    [print(data_item) for data_item in dataset_list]
    print(my_dataset.collate_func(dataset_list))