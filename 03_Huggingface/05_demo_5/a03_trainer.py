import torch, logging
from a01_my_dataset import MyDataset, MyModel
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW

class MyTrainer():

    def __init__(self):
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    # def build_data_loader(self, data_set: MyDataset, batch_size: int):
    #     return DataLoader(
    #         dataset=data_set,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         #舍弃最后一个批次的数据，防止形状出错
    #         drop_last=True,
    #         collate_fn=lambda x: data_set.collate_func(x)
    #     )

    def train(self):
        logger = self.logger
        model = MyModel().to(MyModel.device)

        epochs = 10
        batch_size = 700

        loss_func = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters())

        train_data_set = MyDataset("train")
        train_data_loader = train_data_set.build_data_loader(batch_size)

        val_data_set = MyDataset("validation")
        val_data_loader = val_data_set.build_data_loader(batch_size)

        #初始化最佳验证准确率
        best_val_acc = 0.0

        for epoch in range(epochs):
            model.train()
            for i, (input, target) in enumerate(train_data_loader):
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
                    self.logger.info(f"epoch: {epoch}, i: {i}, loss: {loss.item():4f}, acc: {acc}")
        
            torch.save(model.state_dict(), f"03_Huggingface/05_demo_5/params/{epoch}_bert.pth")
            logger.info("保存成功!")

            model.eval()
            val_acc = 0.0
            val_loss = 0.0
            with torch.no_grad():
                # logger.info("len(train_data_loader): %s}", len(train_data_loader))
                # logger.info("len(val_data_loader): %s", len(val_data_loader))
                for i, (val_input, val_target) in enumerate(val_data_loader):
                        # logger.info("leng(val_target): %s", len(val_target))
                        output = model.forward(val_input)
                        # logger.info("leng(output): %s", len(output))
                        val_loss += loss_func(output, val_target)
                        output = output.argmax(dim=1)
                        # logger.info("leng(output): %s", len(output))

                        # val_loss += val_loss
                        val_acc += (output == val_target).sum().item()
                        # logger.info(f"val_acc: {val_acc}")
                val_loss /= len(val_data_loader)
                val_acc /= (len(val_data_loader) * val_data_loader.batch_size)
                logger.info(f"验证集: loss: {val_loss}, acc: {val_acc}")


if __name__ == "__main__":
    my_trainer = MyTrainer()
    my_trainer.train()