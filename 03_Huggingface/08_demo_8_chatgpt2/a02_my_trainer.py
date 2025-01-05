import sys
sys.path.append(".")
import torch
from a00_my_logging import build_logger
from a00_constant import batch_size, learning_rate, models__uer__gpt2_chinese_cluecorpussmall
from a01_my_dataset import MyDataset
from a01_my_dataset import device
from transformers import AutoModelForCausalLM, GPT2LMHeadModel
from transformers.optimization import get_scheduler
from torch.optim import AdamW

model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(models__uer__gpt2_chinese_cluecorpussmall).to(device)

class MyTrainer():
    def train(self):
        logger = build_logger(__class__.__name__)
        my_data_loader = MyDataset().build_data_loader(batch_size)

        epochs = 3000
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = get_scheduler(
            "linear", optimizer=optimizer, 
            num_training_steps=len(my_data_loader), num_warmup_steps=0
        )

        for epoch in range(epochs):
            for i, input in enumerate(my_data_loader):
                output = model(**input)
                loss = output["loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
                optimizer.step()
                scheduler.step()

                optimizer.zero_grad()
                model.zero_grad()

                if i % 50 == 0:  # 每隔50个批次打印一次信息
                    labels = input["labels"][:, 1:]  # 获取真实标签，忽略<bos>标记
                    output = output["logits"].argmax(dim=2)[:,:-1]  # 获取预测结果，忽略<eos>标记

                    select = labels != 0  # 选择非填充的标签
                    labels = labels[select]  # 应用选择
                    output = output[select]  # 应用选择
                    del select  # 删除不再使用的select
                    # 计算准确率
                    acc = (labels == output).sum().item() / labels.numel()  # 计算准确率的公式
                    lr = optimizer.state_dict()["param_groups"][0]['lr']  # 获取当前学习率

                    # 打印训练信息
                    # print(f"epoch:{epoch},batch:{i},loss:{loss.item()},lr:{lr},acc:{acc}")
                    logger.info(
                        "epoch: %d, batch: %d, loss: %0.4f, lr: %0.5f, acc: %0.4f",
                        epoch, i, loss.item(), lr, acc
                    )

            # 保存最后一轮模型参数
            torch.save(model.state_dict(), "params/net.pt")  # 保存模型参数到指定路径
            print("权重保存成功！")  # 打印成功信息

# 当该脚本作为主程序运行时，调用训练函数
if __name__ == '__main__':
    MyTrainer().train()  # 开始训练过程