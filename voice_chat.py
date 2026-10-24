import requests
import json
import wave
import pyaudio
import tempfile
import os
from faster_whisper import WhisperModel

# ============ 1. 录音函数 ============
def record_audio(duration=5, sample_rate=16000):
    """
    录制麦克风音频，保存为临时WAV文件
    duration: 录音时长（秒）
    sample_rate: 采样率（16000是Whisper标准）
    """
    print(f"🎤 请说话...（{duration}秒录音）")
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024
    )
    
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # 保存为临时WAV文件
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_wav.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    print("✅ 录音完成")
    return temp_wav.name

# ============ 2. 语音转文字（Whisper） ============
def speech_to_text(audio_path):
    """使用 faster-whisper 将音频转为文字"""
    # 使用 tiny 模型（极小，CPU可跑，速度极快）
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    
    segments, _ = model.transcribe(audio_path, language="zh")
    text = "".join([seg.text for seg in segments])
    
    # 删除临时音频文件
    os.unlink(audio_path)
    
    return text.strip()

# ============ 3. 调用Ollama ============
def ask_ollama(prompt_text):
    """向Ollama发送问题，获取回复"""
    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": prompt_text,
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "抱歉，我没有理解您的问题。")
        else:
            return f"请求失败，状态码：{response.status_code}"
    except Exception as e:
        return f"连接Ollama失败：{e}"

# ============ 4. 主流程 ============
if __name__ == "__main__":
    print("=" * 50)
    print("🎙️ 校园语音助手已启动（按Ctrl+C退出）")
    print("=" * 50)
    
    while True:
        try:
            # 步骤1：录音
            audio_file = record_audio(duration=5)
            
            # 步骤2：语音转文字
            user_text = speech_to_text(audio_file)
            if not user_text:
                print("⚠️ 未检测到语音，请重试")
                continue
            print(f"📝 识别结果：{user_text}")
            
            # 步骤3：问Ollama
            reply = ask_ollama(user_text)
            print(f"🤖 助手回复：{reply}\n")
            
        except KeyboardInterrupt:
            print("\n👋 已退出语音助手")
            break
        except Exception as e:
            print(f"❌ 发生错误：{e}")
            continue