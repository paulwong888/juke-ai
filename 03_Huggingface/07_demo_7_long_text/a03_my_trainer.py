import sys
sys.path.append(".")
import torch
from a00_constant import batch_size, learn_rate
from a01_my_data_set import MyDataset
from a02_my_model import MyModel
from a00_my_logging import build_logger
from torch import nn
from torch.optim import AdamW

class MyTrainer():
    def train(self):
        logger = build_logger(__class__.__name__)
        train_data_loader = MyDataset("train").build_data_loader(batch_size)
        val_data_loader = MyDataset("validation").build_data_loader(batch_size)

        my_model = MyModel().instance()

        epochs = 10
        loss_func = nn.CrossEntropyLoss()
        optimizer = AdamW(my_model.parameters())#, lr=learn_rate)

        for epoch in range(epochs):
            my_model.train()
            for i, (input, target) in enumerate(train_data_loader):
                output = my_model.forward(input)
                loss = loss_func(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i%5 == 0:
                    output = torch.argmax(output, dim=1)
                    acc = (output==target).sum().item() / len(target)
                    logger.info(
                        "epoch: %d, i: %d, loss: %0.4f, acc: %0.4f",
                        epoch, i, loss, acc
                    )

            my_model.eval()
            with torch.no_grad():
                for (input, target) in val_data_loader:
                    output = my_model.forward(input)
                    output = torch.argmax(output, dim=1)
                    acc = (output==target).sum().item()

if __name__ == "__main__":
    MyTrainer().train()