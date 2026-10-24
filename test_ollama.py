import requests
import json

# 测试Ollama API是否正常
def test_ollama():
    # 注意：模型名字必须是 qwen2.5:1.5b（你拉取的那个）
    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": "你好，请用一句话介绍一下你自己",
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("✅ Ollama回复：", result.get("response", "无回复"))
        else:
            print("❌ 请求失败，状态码：", response.status_code)
            print("错误信息：", response.text)
    except Exception as e:
        print("❌ 连接失败，请确认Ollama服务已启动：", e)

if __name__ == "__main__":
    test_ollama()