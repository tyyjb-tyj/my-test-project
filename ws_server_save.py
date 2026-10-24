import asyncio
import websockets
import wave
import tempfile
import os
from datetime import datetime

# 音频保存目录
AUDIO_DIR = "received_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

async def handle_esp32(websocket):
    print(f"✅ ESP32 已连接：{websocket.remote_address}")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # 收到二进制音频
                print(f"🎵 收到音频：{len(message)} 字节")
                
                # 保存为WAV文件（假设ESP32发来的是裸PCM，16kHz/16bit/单声道）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_path = os.path.join(AUDIO_DIR, f"audio_{timestamp}.wav")
                
                with wave.open(wav_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)       # 16bit
                    wf.setframerate(16000)   # 16kHz
                    wf.writeframes(message)
                
                print(f"💾 已保存：{wav_path}")
                
                # 预留：这里接 Whisper 识别
                # text = speech_to_text(wav_path)
                # print(f"📝 识别结果：{text}")
                
                # 回复
                await websocket.send(f"音频已收到，{len(message)}字节")
            else:
                # 收到文本
                print(f"📩 收到文本：{message}")
                await websocket.send("文本已收到")
    except websockets.exceptions.ConnectionClosed:
        print("❌ ESP32 已断开")

async def main():
    async with websockets.serve(handle_esp32, "0.0.0.0", 12345, max_size=10*1024*1024):
        print("🚀 WebSocket 服务器已启动")
        print("   监听：0.0.0.0:12345")
        print(f"   音频保存目录：{AUDIO_DIR}")
        print("   等待 ESP32 连接...")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())