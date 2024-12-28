import torch
from a01_my_dataset import MyDataset, MyModel
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW

class MyTrainer():
    def train(self):
        data_set = MyDataset("train")
        model = MyModel()

        epochs = 10
        batch_size = 100

        loss_func = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters())

        data_loader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: data_set.collate_func(x)
        )

        for epoch in range(epochs):
            for i, (input, target) in enumerate(data_loader):
                # print(f"input.shape: {input['input_ids'].shape}")
                # print(f"target.shape: {target.shape}")
                output = model.forward(input)
                # print(f"output.shape: {output.shape}")
                loss = loss_func(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i%5 == 0:
                    output = torch.argmax(output, dim=1)
                    acc = (output == target).sum() / len(target)
                    print(f"epoch: {epoch}, i: {i}, loss: {loss.item()}, acc: {acc}")
        
        torch.save(model.parameters(), r"params/{epoch}_bert.pth")
        print("保存成功!")

if __name__ == "__main__":
    my_trainer = MyTrainer()
    my_trainer.train()