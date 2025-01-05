import os
import pandas as pd
from a00_constant import (
    weibo_train_file_path, weibo_validation_file_path,
    weibo_under_sample_validation_file_path,
    weibo_over_sample_validation_file_path,
    weibo_over_sample_train_file_path
)

class DataTest():
    def test(self):
        print(weibo_under_sample_validation_file_path)
        # weibo_content = pd.read_csv(weibo_train_file_path)bn 
        # weibo_content = pd.read_csv(weibo_validation_file_path)
        # weibo_content = pd.read_csv(weibo_under_sample_validation_file_path)
        # weibo_content = pd.read_csv(weibo_over_sample_validation_file_path)
        weibo_content = pd.read_csv(weibo_over_sample_train_file_path)
        print(weibo_content)
        value_count = weibo_content["label"].value_counts()
        print(value_count)

        total_size = len(weibo_content)
        count_by_percent = (value_count / total_size) * 100
        print(count_by_percent)

if __name__ == "__main__":
    DataTest().test()