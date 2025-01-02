import torch, logging
from a01_my_dataset import MyDataset, MyModel
from torch import nn

class MyModelTester():

    def test(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        my_dataset = MyDataset("test")
        my_data_loader = my_dataset.build_data_loader(700)
        my_model = MyModel().to(MyModel.device)
        my_model.load_state_dict(
            torch.load("03_Huggingface/05_demo_5/params/9_bert.pth", 
                       map_location=MyModel.device)
        )

        running_acc = 0.0
        total_acc = 0.0
        total_size = 0.0
        my_model.eval()
        logger.info("len(my_data_loader.sampler): %s", len(my_data_loader.sampler))
        with torch.no_grad():
            for i, (input, target) in enumerate(my_data_loader):
                output = my_model.forward(input)
                logger.info(
                    "output-size: %s, target-size: %s", 
                    output.shape, target.shape
                )
                logger.info(output)
                output = torch.argmax(output, dim=1)
                logger.info(output)
                logger.info("output-size: %s", output.shape)

                # running_loss += loss
                running_acc = (output == target).sum().item() 
                total_acc += running_acc
                running_acc /= len(target)
                total_size += len(target)
                logger.info("i: %s, acc: %s", i, running_acc)
            logger.info("total acc: %s", total_acc / total_size)

if __name__ == "__main__":
    MyModelTester().test()