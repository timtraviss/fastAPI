# Complete Setup Guide: Dropbox + n8n + Telegram PDF Q&A

## Overview
This setup creates an automated workflow that:
1. **Monitors Dropbox** for new PDF files
2. **Processes PDFs** using your FastAPI service
3. **Enables Telegram chat** with your documents
4. **Sends notifications** when files are processed

## 🚀 Step 1: Setup FastAPI Service

### Install Dependencies
```bash
pip install fastapi uvicorn python-multipart PyMuPDF openai pinecone python-dotenv
```

### Create .env File
```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Run FastAPI Service
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Test at: http://localhost:8000/docs

## 🤖 Step 2: Setup Telegram Bot

### Create Bot
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot`
3. Choose a name and username
4. Save the **Bot Token**

### Get Your Chat ID
1. Message your bot
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Find your `chat.id` in the response

## 📦 Step 3: Setup Dropbox

### Create Dropbox App
1. Go to [