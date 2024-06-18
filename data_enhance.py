from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import LLMChain, HuggingFacePipeline, PromptTemplate
import torch

import csv

model_path = "/data/zhouwz/LLMmodel/Meta-Llama-3-8B-Instruct"
device = torch.device("cuda:2")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) # 加载分词器
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").half() # 加载模型
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    top_p=1,
    repetition_penalty=1.15
)
llama_model = HuggingFacePipeline(pipeline=pipe)
template = '''
# background information # 
Now that you're a writer, you're familiar with synonymous sentence generation in English.  
# questions # 
Provide 10 questions with the same meaning as the following: {original sentence}.
'''
prompt = PromptTemplate(
    input_variables=["original sentence"],
    template=template
)
chain = LLMChain(llm=llama_model, prompt=prompt)
# print(chain.run("What medications or measures can prevent joint symptoms in Ehlers Danlos syndrome patients?"))

instruction_list_1 = []
input_list_1 = []
csv_reader_1 = csv.reader(open("ask_dataset/Hypophosphatasia_dataset.csv"))
for row_1 in csv_reader_1:
	instruction_list_1.append(row_1[0])
	input_list_1.append(row_1[1])
# print("instruction_list_1: \n", instruction_list_1)
# print("input_list_1: \n", input_list_1)

instruction_list_2 = []
input_list_2 = []
csv_reader_2 = csv.reader(open("ask_dataset/Ehlers_Danlos_Syndrome_dataset.csv"))
for row_2 in csv_reader_2:
	instruction_list_2.append(row_2[0])
	input_list_2.append(row_2[1])

instruction_list = [*instruction_list_1[1:], *instruction_list_2[1:]] # 拼接后最后一个元素有个' '（即空格），后面要删除
input_list = [*input_list_1[1:], *input_list_2[1:]]
instruction_list.pop()
input_list.pop()
# print("\n\ninstruction: {} \n\ninstruction: {}.".format(instruction_list, input_list))
print("\n\nthe length of instruction: {}, \n\nthe length of input: {}.".format(len(instruction_list), len(input_list)))


all_expansion_list = []
all_system_list = []

for i in range(len(input_list)):
    print("i = ", i)
    print("input_list: \n", input_list[i])
    output = chain.run(input_list[i])
    # print("output: \n", output)
    split_output = output.split('\n')

    expansion_list = []
    system_list = []
    for j in range(len(split_output)):
        for k in range(1, 11):
            num = str(k) + ". "
            if num in split_output[j]:
                if "?" in split_output[j]:
                    index = split_output[j].find(num)
                    # print("index: ", index)
                    result = split_output[j][index + len(num):]
                    # print("result", result)
                    expansion_list.append(result)
                    system_list.append(instruction_list[i])


    expansion_list.append(input_list[i])
    system_list.append(instruction_list[i])

    for m in range(len(expansion_list)):
        all_expansion_list.append(expansion_list[m])
        all_system_list.append(system_list[m])

all_input_list = [[all_system_list[i],all_expansion_list[i]] for i in range(len(all_expansion_list))]

with open('enhanced_ask_dataset/system_input.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(all_input_list)