# 导⼊操作系统相关模块，⽤于设置环境变量
import os

# 设置⽤⼾代理，模拟浏览器请求头，避免某些⽹站拒绝爬⾍访问
os.environ[
 'USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

# 导⼊Chroma向量数据库
import chromadb
# 从langchain_ollama导⼊OllamaLLM，⽤于加载本地Ollama部署的⼤模型
from langchain_ollama import OllamaLLM
# 导⼊bs4（BeautifulSoup），⽤于解析⽹⻚内容
import bs4
# 从langchain社区⼯具导⼊WebBaseLoader，⽤于加载⽹⻚⽂档
from langchain_community.document_loaders import WebBaseLoader
# 导⼊⽂本分割器，⽤于将⻓⽂档分割成⼩块
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 从langchain_ollama导⼊OllamaEmbeddings，⽤于⽣成⽂本嵌⼊向量
from langchain_ollama import OllamaEmbeddings
# 导⼊提⽰词模板类
from langchain_core.prompts import PromptTemplate
# 导⼊检索增强⽣成（RAG）的问答链
from langchain.chains import RetrievalQA
# 从langchain_chroma导⼊Chroma向量存储集成
from langchain_chroma import Chroma # 这⾥需要安装包：pip install langchain_chroma

def lang_rag():
# 1. 初始化⼤语⾔模型(LLM)
    llm = OllamaLLM(
        model="deepseek-r1", # 指定使⽤的模型名称
        temperature=0.1, # 温度参数，控制输出的随机性，值越低越确定
        top_p=0.4 # 核采样参数，控制输出的多样性
# callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]) # 可选：流式输出回调
    )

# 2. 加载⽹⻚⽂档
# 使⽤WebBaseLoader加载指定⽹⻚内容
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",), # 要加载的⽹⻚URL
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer( # 只解析⽹⻚中指定类别的元素，提⾼效率
                class_=("post-content", "post-title", "post-header")
# 要保留的⽹⻚元素类别
            )
        ),
    )
    docs = loader.load() # 执⾏加载，获取⽂档内容

# 3. ⽂本分割
# 初始化⽂本分割器，设置分割参数
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # 每个⽂本块的⼤⼩（字符数）
        chunk_overlap=200 # ⽂本块之间的重叠部分（字符数），⽤于保持上下⽂连贯性
    )
    splits = text_splitter.split_documents(docs) # 对⽂档进⾏分割处理

# 4. 创建向量数据库
    vectorstore = Chroma.from_documents(
        documents=splits, # 要存⼊向量库的⽂本块
        embedding=OllamaEmbeddings(model="nomic-embed-text"), # 使⽤的嵌⼊模型
        collection_name='ddd' # 向量库集合的名称
    )

# 5. 定义提⽰词模板
    prompt = PromptTemplate(
        input_variables=['context', 'question'], # 模板中的变量：上下⽂和问题
        template="""You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the que
        stion.
        If you don't know the answer, just say you don't know without a
        ny explanation.
        Question: {question}
        Context: {context}
        Answer:""", # 提⽰词内容，指导模型如何使⽤上下⽂回答问题
    )
    print('提⽰词模板创建')

    # 6. 创建向量数据库检索器
    retriever = vectorstore.as_retriever() # 将向量库转换为检索器
    print('向量数据库检索器创建')

    # 7. 创建检索增强⽣成(RAG)问答链
    qa_chain = RetrievalQA.from_chain_type(
    llm, # 使⽤的⼤语⾔模型
    retriever=retriever, # 使⽤的检索器
    chain_type_kwargs={"prompt": prompt} # 传⼊提⽰词模板
    )

    # 8. 测试问答功能
    # 第⼀个问题：关于AI Agent
    question = "what is Ai agent?"
    result = qa_chain.invoke({"query": question}) # 调⽤问答链获取答案
    print("question1:")
    print(result)

    # 第⼆个问题：关于React（测试模型对⽆关内容的处理）
    question2 = "what is react?"
    result = qa_chain.invoke({"query": question2})
    print("question2:")
    print(result)

    # 第三个问题：关于智能体（测试中⽂问答能⼒）
    question3 = "什么是智能体?，⽤中⽂回答"
    result = qa_chain.invoke({"query": question3})
    print("question3:")
    print(result)


# 程序⼊⼝：当脚本直接运⾏时执⾏lang_rag()函数
if __name__ == '__main__':
    lang_rag()