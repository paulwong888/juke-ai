import pandas as pd

df = pd.read_csv("data/news/train.csv")
#统计每个类别的数据量
category_counts = df["label"].value_counts()

#统计每个类别的比值
total_data = len(df)
category_ratios = (category_counts / total_data) *100

print(category_counts)
print(category_ratios)