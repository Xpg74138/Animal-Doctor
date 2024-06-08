import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate
from lmdeploy import pipeline, TurbomindEngineConfig
import os

# 加载 RAG 模型
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="/root/Animal-Doctor/data/model/sentence-transformer")
    persist_directory = 'data_base/vector_db/chroma'
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    llm = InternLM_LLM(model_path="/root/Animal-Doctor/data/model/internlm2-chat-20b")
    template = """请列出牛常见的疾病，并提供每种疾病的详细信息。
                  具体需求如下：
                  1. 疾病名称
                  2. 主要症状
                  3. 预防措施
                  4. 治疗方法
                  希望这些信息能帮助更好地了解和管理牛的健康问题。谢谢！
                  ···
                  {context}
                  ···
                  用户的问题: {question}
                  你给的回答:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa_chain

# 加载多模态模型
backend_config = TurbomindEngineConfig(session_len=8192)
pipe = pipeline('/share/new_models/liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config)

class ModelCenter:
    def __init__(self):
        self.chain = load_chain()
    
    def multimodal_answer_with_rag(self, image, text, chat_history: list = []):
        if image is not None:
            multimodal_response = pipe((text, image)).text
            question = multimodal_response  # 将多模态模型的输出作为RAG模型的输入问题
        else:
            question = text
        
        if not question:
            return "", chat_history
        
        try:
            rag_response = self.chain({"query": question})["result"]
            chat_history.append((text, rag_response))
        except Exception as e:
            chat_history.append((text, str(e)))
        
        return "", chat_history

model_center = ModelCenter()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            gr.Markdown("""<h1><center>Animal-Doctor</center></h1><center>动物医生</center>""")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            msg = gr.Textbox(label="Prompt/问题")
            image_input = gr.Image(type="pil", label="图片输入")

            with gr.Row():
                submit_btn = gr.Button("提问")
            with gr.Row():
                clear = gr.ClearButton(components=[chatbot], value="清除")
                
        submit_btn.click(model_center.multimodal_answer_with_rag, inputs=[image_input, msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。<br>
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。<br>""")
    
gr.close_all()
demo.launch()
