import torch
from a00_constant import (
    weibo_under_sample_file_dir,
    weibo_over_sample_file_dir
)
from a01_my_data_set import MyDataset
from a02_my_model import MyModel
from a00_constant import buid_logger
from torch import nn
from torch.optim import AdamW

class MyTrainer():

    def train(self):
        logger = buid_logger()
        train_batch_size = 700
        val_batch_size = 100
        train_data_loader = MyDataset(weibo_over_sample_file_dir, "train").build_data_loader(train_batch_size)
        val_data_loader = MyDataset(weibo_under_sample_file_dir, "validation").build_data_loader(val_batch_size)
        logger.info(len(train_data_loader))
        logger.info(len(val_data_loader))
        my_model = MyModel().get_instance()

        epochs = 10
        loss_func = nn.CrossEntropyLoss()
        optimizer = AdamW(my_model.parameters())

        running_loss = 0.0
        running_acc = 0.0
        for epoch in range(epochs):
            my_model.train()
            for i, (input, target) in enumerate(train_data_loader):
                output = my_model.forward(input)
                loss = loss_func(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss
                if i%5 == 0:
                    output = torch.argmax(output, dim=1)
                    total_acc = (output==target).sum().item()

                    logger.info(
                        "epoch: %s, i: %d, loss: %0.4f, acc: %0.4f, batch_size: %d",
                        epoch, i, loss.item(), total_acc / len(target), train_data_loader.batch_size
                    )
            
            torch.save(my_model.state_dict(), f"03_Huggingface/06_demo_6_custom_vacob/params/{epoch}_bert.pth")
            logger.info("保存成功")

            my_model.eval()
            with torch.no_grad():
                total_acc = 0.0
                total_size = 0
                for (input, target) in val_data_loader:
                    output = my_model.forward(input)
                    output = torch.argmax(output, dim=1)

                    total_acc += (output==target).sum().item()
                    total_size += len(target)
                logger.info("验证集: acc: %0.4f", total_acc / total_size)


if __name__ == "__main__":
    MyTrainer().train()
