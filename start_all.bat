@echo off
echo 正在启动 MySQL...
net start MySQL80

echo 正在启动 Ollama...
start "" ollama serve

echo 等待3秒...
timeout /t 3 /nobreak

echo 启动语音助手...
cd /d D:\voice_assistant_project
python campus_assistant.py

pause