长文本训练要点:
由于改了max_length, 即修改了原模型, 则原先的模型不能用torch.no_grad(), 必须参与训练