import logging
import os

weibo_file_dir = "03_Huggingface/data/Weibo"

weibo_over_sample_file_dir = os.path.join(weibo_file_dir, "over-sampled")

weibo_under_sample_file_dir = os.path.join(weibo_file_dir, "under-sampled")

weibo_train_file_path = os.path.join(weibo_file_dir, "train.csv")

weibo_validation_file_path = os.path.join(weibo_file_dir, "validation.csv")

weibo_under_sample_validation_file_path = os.path.join(weibo_file_dir, "under-sampled", "validation.csv")

weibo_under_sample_train_file_path = os.path.join(weibo_file_dir, "under-sampled", "train.csv")

weibo_over_sample_validation_file_path = os.path.join(weibo_over_sample_file_dir, "validation.csv")

weibo_over_sample_train_file_path = os.path.join(weibo_over_sample_file_dir, "train.csv")

bert_base_chinese_model_path = "03_Huggingface/data/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"

def buid_logger():
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger
