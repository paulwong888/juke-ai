import torch
from a00_constant import poems_file_path, models__uer__gpt2_chinese_cluecorpussmall
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.my_data_set = load_dataset("text", data_files=poems_file_path, split="train")
        self.tokenizer = AutoTokenizer.from_pretrained(models__uer__gpt2_chinese_cluecorpussmall)

    def __len__(self):
        return len(self.my_data_set)

    def __getitem__(self, index):
        return self.my_data_set[index]
    
    def collate_fn(self, data: list[dict]):
        input_list = [item["text"] for item in data]

        input_encoded_dict = self.tokenizer.batch_encode_plus(
            input_list, max_length=1024, truncation=True,
            padding="max_length", return_tensors="pt"
        )

        for key in input_encoded_dict.keys():
            input_encoded_dict[key] = input_encoded_dict[key].to(device)

        input_encoded_dict["labels"] = input_encoded_dict["input_ids"]

        return input_encoded_dict
    
    def build_data_loader(self, batch_size: int):
        return DataLoader(
            self, batch_size=batch_size, shuffle=True,
            drop_last=True,
            collate_fn=lambda data: self.collate_fn(data)
        )
    
    def check_data(self):
        size = 5
        short_data_list = self.my_data_set.select(range(size))
        print(type(short_data_list[0]))
        [print(item) for item in short_data_list]
        print(self.collate_fn(short_data_list))
    
if __name__ == "__main__":
    my_data_set = MyDataset()
    my_data_set.check_data()
