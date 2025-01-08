import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from a00_constant import max_length, repoid_uer_gpt2_chinese_cluecorpussmall, fine_turn_uer_gpt2_chinese_cluecorpussmall

class PoemRunner():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(repoid_uer_gpt2_chinese_cluecorpussmall)
        self.model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(repoid_uer_gpt2_chinese_cluecorpussmall)
        # self.model.load_state_dict(torch.load(fine_turn_uer_gpt2_chinese_cluecorpussmall))

    def predict(self, input_text: str, max_output: int):
        input_dict = self.tokenizer(
            input_text, 
            return_tensors="pt"
        )["input_ids"]
        output = self.model.generate(
            input_dict, 
            do_sample=True,
            max_new_tokens = max_output, 
            # num_return_sequences = 1
        )

        # output_decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # print(output_decoded)
        output_decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        # print(output_decoded)

        return output_decoded

if __name__ == "__main__":
    poem_runner = PoemRunner()
    # output_ori = poem_runner.predict()
    input_text = "床前明月光"
    max_output = 50
    output = poem_runner.predict(input_text, max_output)
    print(output)
    # print(output_argmax.shape)