from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import CTransformers
from src.helper import *

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

instruction = "Convert the following text from English to Spanish: \n\n {text}"

SYSTEM_PROMPT = B_SYS + CUSTOM_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instruction + E_INST

prompt = PromptTemplate(template=template, input_variables=["text"])

llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q4_1.bin',
                    model_type='llama',
                    config={'max_new_tokens': 128,
                            'temperature': 0.01}
                   )

LLM_Chain=LLMChain(prompt=prompt, llm=llm)

print(LLM_Chain.invoke("Good night"))
