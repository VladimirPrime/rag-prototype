from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
import torch
from huggingface_hub import login
from dotenv import load_dotenv
import os
import logging
from cachetools import cached, TTLCache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime

# --- Конфигурация ---
load_dotenv()

# Инициализация Flask
app = Flask(__name__)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Лимитер запросов (100 в час на IP)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# Кэширование ответов (1 час, макс 1000 записей)
cache = TTLCache(maxsize=1000, ttl=3600)


# --- Инициализация модели ---
def load_model():
    try:
        login(token=os.getenv("HUGGING FACE"))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Используется устройство: {device}")

        model_name = "mistralai/Mistral-7B-v0.1"  # Более подходящая версия для диалога
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )

        pipeline_config = {
            "model": model,
            "tokenizer": tokenizer,
            "max_new_tokens": 150,
            "temperature": 0.7,
            "do_sample": True,
            "top_k": 50
        }

        return pipeline("text-generation", **pipeline_config)

    except Exception as e:
        logging.error(f"Ошибка загрузки модели: {str(e)}")
        raise


# --- База знаний ---
def init_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    documents = [
        "RAG (Retrieval-Augmented Generation) - это гибридный подход, сочетающий поиск информации и генерацию текста.",
        "Mistral-7B - 7-миллиардная языковая модель с открытым исходным кодом.",
        "Для работы RAG нужны: 1) Векторная база данных 2) Модель генерации 3) Система поиска."
    ]

    return FAISS.from_texts(documents, embeddings)


# --- Инициализация ---
try:
    model_pipeline = load_model()
    llm = HuggingFacePipeline(pipeline=model_pipeline)
    vectorstore = init_vectorstore()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
except Exception as e:
    logging.critical(f"Ошибка инициализации: {str(e)}")
    exit(1)


# --- Роуты ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/ask', methods=['POST'])
@limiter.limit("10 per minute")  # Лимит для API
@cached(cache)
def ask_question():
    start_time = datetime.now()

    try:
        data = request.get_json()
        question = data.get('question', '').strip()

        if not question or len(question) > 1000:
            logging.warning(f"Некорректный вопрос: {question[:100]}...")
            return jsonify({"error": "Question must be 1-1000 chars"}), 400

        logging.info(f"Обработка вопроса: {question[:50]}...")

        result = qa_chain({"query": question})

        response = {
            "answer": result["result"],
            "sources": [doc.page_content[:100] + "..." for doc in result["source_documents"]],
            "time_ms": (datetime.now() - start_time).total_seconds() * 1000
        }

        logging.info(f"Ответ сгенерирован за {response['time_ms']:.0f} мс")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Ошибка: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# Health check для мониторинга
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model": "Mistral-7B-Instruct",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })


# --- Запуск ---
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)