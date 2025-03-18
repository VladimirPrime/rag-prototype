# RAG Prototype with Mistral 7B

Этот проект представляет собой прототип RAG (Retrieval-Augmented Generation) с использованием модели Mistral 7B. Приложение позволяет задавать вопросы и получать ответы, используя базу знаний и генерацию текста.

## Установка

1. Убедитесь, что у вас установлены:
   - Python 3.9 
   - Docker (позже)

2. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/ваш_username/rag-prototype.git
   cd rag-prototype

3. Установите зависимости:
   pip install -r requirements.txt
4. ## Настройка токенов

1) Создайте файл `.env` в корневой папке проекта.
2) Добавьте в него ваш токен Hugging Face:
   ```plaintext
   HUGGINGFACE_TOKEN=ваш_токен
   
Как получить токен Hugging Face
Перейдите на Hugging Face.

Авторизуйтесь или создайте аккаунт.

Перейдите в настройки профиля: Settings → Access Tokens.

Создайте новый токен и скопируйте его.

5. Запустите приложение:
    python app.py

http://localhost:5000

Зависимости
Flask

Transformers

LangChain

FAISS

Torch