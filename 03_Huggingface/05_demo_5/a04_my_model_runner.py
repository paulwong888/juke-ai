import torch
from a02_my_model import MyModel
from transformers import BertModel, AutoTokenizer

class MyModelRunner():
    def __init__(self, state_dict_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.names = ["正向评价", "负向评价"]
        self.my_model = MyModel().to(self.device)
        # print(self.my_model)
        self.my_model.load_state_dict(torch.load(state_dict_path, map_location=self.device))
        self.tokenizer = AutoTokenizer.from_pretrained(MyModel.model_name)
        self.my_model.eval()

    def predict(self, input_text: str):
        input_text_ids = self.tokenizer.batch_encode_plus(
            [input_text], max_length=512, padding="max_length",
            return_tensors="pt", truncation=True
        ).to(self.device)
        with torch.no_grad():
            output = self.my_model.forward(input_text_ids)
        output = torch.argmax(output, dim=1)

        return self.names[output]
    
if __name__ == "__main__":
    my_model_runner = MyModelRunner("05_demo_5/params/2_bert.pth")
    while True:
        input_text = input("请输入测试数据(输入'q'退出): ")
        if input_text == "q":
            print("测试结束")
            break
        output = my_model_runner.predict(input_text)
        print("模型判定: ", output, "\n")