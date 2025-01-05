import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from a00_constant import news_file_dir, max_length, bert_model_path
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class MyDataset(Dataset):

    def __init__(self, split):
        self.my_dataset = load_dataset("csv", data_dir=news_file_dir, split=split)
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(bert_model_path)

    def __len__(self):
        return len(self.my_dataset)

    def __getitem__(self, item):
        return self.my_dataset[item]

    def check_data(self):
        print(self)
        short_dataset = self.my_dataset.select(range(200))
        # [print(type(item)) for item in short_dataset]
        [print(len(item["text"])) for item in short_dataset]
        # print(short_dataset["text"]) # 提取 text这一列的数据,类型为list
        my_df = self.my_dataset.to_pandas() #DataFrame
        text_series = my_df["text"] #Series
        # print(my_df["label"].value_counts())
        print(type(my_df))
        print(type(text_series))
        # longest_item = my_df["text"].apply(len).idxmax()
        # longest_value = my_df.loc[longest_item, "text"]
        # print(len(my_df.loc[my_df["text"].apply(len).idxmax(), "text"]))
        print(f"最长的文本的长度是: {len(text_series[text_series.apply(len).idxmax()])}")
        print(f"最短的文本的长度是: {len(text_series[text_series.apply(len).idxmin()])}")

    
    def collate_fn(self, data_list: list[dict]):
        """
        data_list的类型是,以dict为单位的list
        """
        input_list = [data["text"] for data in data_list]
        target_list = [data["label"] for data in data_list]

        input_encoded_dict = self.tokenizer.batch_encode_plus(
            input_list, max_length=max_length,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        # print(input_encoded_dict["input_ids"].shape)
        for key in input_encoded_dict.keys():
            input_encoded_dict[key] = input_encoded_dict[key].to(device)
        target_tensor = torch.LongTensor(target_list).to(device)

        return input_encoded_dict, target_tensor
    
    def build_data_loader(self, batch_size):
        return DataLoader(
            self, batch_size=batch_size, drop_last=True,
            shuffle=True, collate_fn=lambda data_list: self.collate_fn(data_list)
        )

if __name__ == "__main__":
    my_dataset = MyDataset("train")
    my_dataset.check_data()
    data_list = my_dataset.my_dataset.select(range(2))
    print(my_dataset.collate_fn(data_list))