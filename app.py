import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import subprocess
import sys
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate
import gc

# Modify chromadb __init__.py for sqlite3 compatibility
chromadb_init_path = "/usr/local/share/python/.pyenv/versions/3.10.13/lib/python3.10/site-packages/chromadb/__init__.py"
replacement_code = """
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
"""

try:
    with open(chromadb_init_path, "r") as file:
        content = file.read()

    if replacement_code not in content:
        with open(chromadb_init_path, "w") as file:
            file.write(replacement_code + content)
        print("Successfully modified chromadb __init__.py")
    else:
        print("chromadb __init__.py already modified")
except FileNotFoundError:
    print(f"File not found: {chromadb_init_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# Download models only if they are not already present
base_path = 'Animal-Doctor/data/model/internlm2-chat-1_8b'
if not os.path.exists(base_path):
    os.system(f'git clone git@code.openxlab.org.cn:comefly/internlm-chat-1.8b.git {base_path}')
    os.system(f'cd {base_path} && git lfs install && git lfs pull')

# Download Sentence Transformer
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sentence_transformer_path = 'Animal-Doctor/data/model/sentence-transformer'
if not os.path.exists(sentence_transformer_path):
    os.system(f'huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir {sentence_transformer_path}')

# Load chain function
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name=sentence_transformer_path)

    persist_directory = 'Animal-Doctor/data_base/vector_db/chroma'

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    llm = InternLM_LLM(model_path="Animal-Doctor/data/model/internlm2-chat-1_8b")

    template = """请列出畜禽常见的疾病，并提供每种疾病的详细信息。

                具体需求如下：
                1. 疾病名称
                2. 主要症状
                3. 预防措施
                4. 治疗方法

                希望这些信息能帮助更好地了解和管理畜禽的健康问题。谢谢！
    ···
    {context}
    ···
    用户的问题: {question}
    你给的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                     template=template)

    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever(),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    return qa_chain

class Model_center():
    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, env_temp: float, animal_temp: float, humidity: float, feed_intake: float, particle_concentration: float, ammonia_concentration: float, chat_history: list = []):
        if question is None or len(question) < 1:
            return "", chat_history
        try:
            context = {
                "env_temp": env_temp,
                "animal_temp": animal_temp,
                "humidity": humidity,
                "feed_intake": feed_intake,
                "particle_concentration": particle_concentration,
                "ammonia_concentration": ammonia_concentration,
            }
            context_str = '\n'.join([f"{key}: {value}" for key, value in context.items()])
            prompt = f"上下文信息: {context_str}\n问题: {question}"
            result = self.chain({"query": prompt})["result"]
            chat_history.append((question, result))
            gc.collect()  # Manually trigger garbage collection
            return "", chat_history
        except Exception as e:
            return str(e), chat_history

model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            gr.Markdown("""<h1><center>Animal-Doctor</center></h1>""")

    with gr.Row():
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(height=550, show_copy_button=True)
        with gr.Column(scale=5):
            env_temp = gr.Number(label="环境温度/℃", value=25.0)
            humidity = gr.Number(label="环境湿度/%", value=60.0)
            particle_concentration = gr.Number(label="颗粒物浓度/微克/m³", value=2.0)
            ammonia_concentration = gr.Number(label="氨气浓度/ppm", value=8.0)
            animal_temp = gr.Number(label="畜禽体温/℃", value=38.5)
            feed_intake = gr.Number(label="平均饲料摄入量/KG", value=2.0)
    with gr.Row():
        msg = gr.Textbox(label="问题")
    with gr.Row():
        submit_btn = gr.Button("提交")
    with gr.Row():
        clear = gr.ClearButton(components=[chatbot], value="清除")

    submit_btn.click(model_center.qa_chain_self_answer, 
                     inputs=[msg, env_temp, animal_temp, humidity, feed_intake, particle_concentration, ammonia_concentration, chatbot], 
                     outputs=[msg, chatbot])
    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示。 <br>
    """)
# Clean up old Gradio sessions
gr.close_all()
demo.launch()

