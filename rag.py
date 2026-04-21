from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. PDFを読み込む
print("PDFを読み込み中...")
loader = PyPDFLoader("market_analysis.pdf")
documents = loader.load()

# 2. チャンクに分割する
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"チャンク数: {len(chunks)}")

# 3. ベクトルDBに保存する
print("ベクトルDBに保存中...")
embeddings = OllamaEmbeddings(model="llama3.2", base_url="http://172.26.16.1:11434", client_kwargs={"timeout": 300})
db = Chroma.from_documents(chunks, embeddings)

# 4. 質問する
question = "日本食特化レシピAPIの市場規模はどのくらいですか？"
print(f"\n質問: {question}")

docs = db.similarity_search(question, k=3)
context = "\n".join([d.page_content for d in docs])

# 4. 対話式で質問する
llm = OllamaLLM(model="llama3.2", base_url="http://172.26.16.1:11434", client_kwargs={"timeout": 300})

print("\n質問してください（終了するには 'quit' と入力）")
while True:
    question = input("\n質問: ")
    if question.lower() == "quit":
        break

    docs = db.similarity_search(question, k=5)
    for i, doc in enumerate(docs):
        print(f"\n--- チャンク{i+1} ---")
        print(doc.page_content)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""以下の文章を参考に質問に答えてください。

{context}

質問: {question}
"""
    print("\n回答:")
    print(llm.invoke(prompt))