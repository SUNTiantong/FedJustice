import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7B", device_map="auto", trust_remote_code=True)

peft_model = PeftModel.from_pretrained(model, 'FedJudge/fedjudge-base-7b', torch_dtype=torch.float32).half()
# peft_model = PeftModel.from_pretrained(model, 'FedJudge/fedjudge-cl-7b',torch_dtype=torch.float32).half()
# peft_model = PeftModel.from_pretrained(model, 'FedJudge/fedjudge-cl-client3-7b',torch_dtype=torch.float32).half()

data = '假设你是一名律师，请回答以下向你咨询的问题：在法律中定金与订金的区别是什么？'

inputs = tokenizer(data, return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = peft_model.generate(**inputs, max_new_tokens=500, repetition_penalty=1.1)

pred_result = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
print(pred_result.split(data)[-1])