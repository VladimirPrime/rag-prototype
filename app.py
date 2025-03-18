from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
import torch
from huggingface_hub import login
from dotenv import load_dotenv
import os

# Загружаем переменные из .env
load_dotenv()

# Авторизация в Hugging Face через токен из .env
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Инициализация Flask
app = Flask(__name__)

# Проверяем, что CUDA доступна
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# Загружаем модель и токенизатор Mistral 7B с 4-битным квантованием
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # 4-битное квантование
    device_map="auto"  # Автоматически распределяет нагрузку на GPU/CPU
)

# Создаём pipeline для модели
model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,  # Увеличиваем количество генерируемых токенов
    temperature=0.7,  # Управляем "креативностью" модели
    do_sample=True,  # Включаем случайную генерацию
)

# Оборачиваем pipeline в HuggingFacePipeline для LangChain
llm = HuggingFacePipeline(pipeline=model_pipeline)

# Загружаем базу знаний (например, тексты экспертных ответов)
documents = [
    "Ответ 1: RAG — это подход, который объединяет поиск и генерацию текста. Он использует базу знаний для улучшения ответов модели.",
    "Ответ 2: RAG состоит из двух этапов: поиск релевантной информации в базе знаний и генерация ответа на основе найденной информации.",
    "Ответ 3: RAG позволяет моделям генерировать более точные и информативные ответы, используя внешние источники данных.",
    # Добавьте сюда больше данных
]

# Создаём векторное хранилище для поиска
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Указываем модель явно
vectorstore = FAISS.from_texts(documents, embeddings)

# Настраиваем RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  # Используем обёрнутую модель
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)


# Маршрут для отображения главной страницы
@app.route('/')
def index():
    return render_template('index.html')


# Маршрут для обработки вопросов
@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Получаем вопрос из запроса
        question = request.json.get('question')
        if not question:
            return jsonify({'error': 'Question is required'}), 400

        # Генерация ответа с помощью RAG и Mistral 7B
        answer = qa_chain.run(question)
        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Запуск Flask-приложения
if __name__ == '__main__':
    app.run(debug=True)