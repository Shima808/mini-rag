# Mini RAG Demo 🔍

**PDF質問応答システム — Built with LangChain + ChromaDB + Ollama**

## 概要

PDFドキュメントを読み込み、自然言語で質問できるRAG（Retrieval-Augmented Generation）システム。
LLMの学習データに含まれない社内文書や独自資料に対して質問応答が可能。

## 技術構成

- **LangChain** — RAGパイプラインの構築
- **ChromaDB** — ベクトルデータベース（ローカル）
- **Ollama** — ローカルLLM実行環境（llama3.2）
- **PyPDF** — PDFテキスト抽出

## セットアップ

### 1. Ollamaのインストール・起動

https://ollama.com からインストール後：

    ollama pull llama3.2
    ollama serve

### 2. 仮想環境の作成

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### 3. 実行

    python3 rag.py

## 工夫した点

- chunk_size=800 / overlap=100 にチューニングし、文章の途中切れによる検索精度低下を解消
- ローカルLLMを採用し、機密文書を外部APIに送信せずに処理できる構成に
- WSL環境からWindowsのOllamaサーバーに接続する構成で動作確認済み
