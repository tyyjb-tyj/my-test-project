import requests
import json
import wave
import pyaudio
import tempfile
import os
import webbrowser
import numpy as np
from faster_whisper import WhisperModel
import pymysql
from sentence_transformers import SentenceTransformer
import chromadb
from zhconv import convert

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==========================================
# 1. 配置
# ==========================================
CAMPUS_ACTIONS = {
    0: {"name": "查课表", "url": "https://www.hmlc.edu.cn/"},
    1: {"name": "申请请假", "url": "http://xsc.hmlc.edu.cn/"},
    2: {"name": "人脸采集", "url": "https://www.hmlc.edu.cn"},
    3: {"name": "查成绩", "url": "https://www.hmlc.edu.cn/"},
    4: {"name": "校园官网", "url": "https://www.hmlc.edu.cn"},
}

DB_CONFIG = {
    'host': 'localhost', 'user': 'root', 'password': '123456',
    'database': 'hmlc_college', 'charset': 'utf8mb4'
}

SIMILARITY_THRESHOLD = 0.55
WHISPER_PROMPT = "校园助手"
EXIT_WORDS = ['退出', '再见', '结束', '拜拜', '关闭', '停止']

# ==========================================
# 2. 初始化
# ==========================================
print("🔧 正在加载嵌入模型...")
embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
print("✅ 嵌入模型加载完成")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="faq", metadata={"hnsw:space": "cosine"}
)

print("🔧 正在加载 Whisper 模型...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
print("✅ Whisper 模型加载完成")

# ==========================================
# 3. 语音转文字
# ==========================================
def speech_to_text(audio_path):
    segments, _ = whisper_model.transcribe(
        audio_path, language="zh",
        initial_prompt=WHISPER_PROMPT, vad_filter=True
    )
    text = "".join([seg.text for seg in segments])
    os.unlink(audio_path)
    return convert(text, 'zh-cn').strip()

# ==========================================
# 4. 构建/更新向量索引
# ==========================================
def build_vector_index():
    if collection.count() > 0:
        print(f"✅ 向量索引已存在，共 {collection.count()} 条数据")
        return
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, question, answer FROM faq")
    faqs = cursor.fetchall()
    conn.close()
    if not faqs:
        print("⚠️ 数据库中没有 FAQ 数据")
        return
    for id, question, answer in faqs:
        combined = f"{question} {answer}"
        emb = embedding_model.encode(combined).tolist()
        collection.add(
            documents=[combined], embeddings=[emb],
            metadatas=[{"question": question, "answer": answer}],
            ids=[str(id)]
        )
    print(f"✅ 向量索引构建完成，共 {len(faqs)} 条数据")

# ★ 改动3：知识库热更新（运行时加FAQ，无需重建索引）
def add_faq_hot(question, answer, category='自定义'):
    """向数据库和向量索引同时添加FAQ，热更新"""
    try:
        # 1. 写入数据库
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO faq (question, answer, category) VALUES (%s, %s, %s)",
            (question, answer, category)
        )
        conn.commit()
        new_id = cursor.lastrowid
        conn.close()

        # 2. 同步到向量索引
        combined = f"{question} {answer}"
        emb = embedding_model.encode(combined).tolist()
        collection.add(
            documents=[combined], embeddings=[emb],
            metadatas=[{"question": question, "answer": answer}],
            ids=[str(new_id)]
        )
        print(f"✅ 热更新成功: {question}")
        return True
    except Exception as e:
        print(f"❌ 热更新失败: {e}")
        return False

build_vector_index()

# ==========================================
# 5. 向量检索 + SQL 兜底
# ==========================================
def query_faq_vector(user_question):
    query_emb = embedding_model.encode(user_question).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=1)

    if results['distances'] and results['distances'][0]:
        distance = results['distances'][0][0]
        similarity = 1 - distance
        print(f"🔍 余弦距离: {distance:.4f}, 相似度: {similarity:.4f}")
        if similarity >= SIMILARITY_THRESHOLD:
            matched_q = results['metadatas'][0][0]['question']
            print(f"✅ 向量检索命中: {matched_q}")
            return results['metadatas'][0][0]['answer']

    print("🔍 向量检索未命中，尝试 SQL LIKE 匹配...")
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    keyword = None
    for kw in ['请假', '课表', '人脸', '成绩', '官网', '走读',
               '图书馆', '食堂', '宿舍', '学费', '招生', '地址',
               '电话', '专业', '学院', '校历', '一卡通', '宿舍']:
        if kw in user_question:
            keyword = kw
            break
    if keyword is None:
        keyword = user_question

    pattern = f"%{keyword}%"
    cursor.execute("SELECT question, answer FROM faq WHERE question LIKE %s LIMIT 1", (pattern,))
    result = cursor.fetchone()
    if not result:
        cursor.execute("SELECT question, answer FROM faq WHERE answer LIKE %s LIMIT 1", (pattern,))
        result = cursor.fetchone()
    conn.close()

    if result:
        print(f"✅ SQL LIKE 命中: {result[0]}")
        return result[1]
    return None

# ==========================================
# 6. VAD 动态录音
# ==========================================
def record_audio_vad(max_duration=10, sample_rate=16000, silence_threshold=800):
    print("🎤 请说话...（说完自动停止）")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                    input=True, frames_per_buffer=1024)
    frames, silent_chunks, speaking = [], 0, False
    max_silent = int(sample_rate / 1024 * 1.5)

    for _ in range(int(sample_rate / 1024 * max_duration)):
        data = stream.read(1024)
        audio_data = np.frombuffer(data, dtype=np.int16)
        volume = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        if volume > silence_threshold:
            speaking = True
            silent_chunks = 0
            frames.append(data)
        elif speaking:
            silent_chunks += 1
            frames.append(data)
            if silent_chunks >= max_silent:
                break

    stream.stop_stream(); stream.close(); p.terminate()

    if not frames or len(frames) < 15:
        print("⚠️ 未检测到有效语音")
        return None

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_wav.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    print("✅ 录音完成")
    return temp_wav.name

# ==========================================
# 7. Qwen 意图识别（★ 改动1：带多轮对话记忆）
# ==========================================
def ask_campus_ai(user_text, chat_history=None):
    # 把最近3轮历史拼进Prompt
    history_text = ""
    if chat_history:
        recent = chat_history[-3:]
        history_text = "\n".join([f"用户: {h['user']}\n助手: {h['bot']}" for h in recent])
        history_text = f"\n【最近对话历史】\n{history_text}\n"

    prompt = f"""你是校园APP语音指令解析器。{history_text}
**规则**：
1. 只有用户**明确表达执行意图**（如"我要请假"、"打开课表"）才输出对应动作ID。
2. 询问句（含"怎么、如何、什么、哪里"）或无关内容 → 输出 action_id = 5。

功能ID：
0 - 查课表（含"课表"）
1 - 申请请假（含"请假"）
2 - 人脸采集（含"人脸"）
3 - 查成绩（含"成绩"）
4 - 校园官网（含"官网"）
5 - 不知道/无关

**同音词容忍**："特表/差课表/茶课表"→查课表；"请驾/亲家"→请假。

用户说："{user_text}"

只输出JSON，如：{{"action_id": 5, "reply": "抱歉，我听不懂"}}
"""
    payload = {"model": "qwen2.5:1.5b", "prompt": prompt, "stream": False}
    try:
        response = requests.post("http://localhost:11434/api/generate",
                                 json=payload, timeout=30)
        if response.status_code == 200:
            raw = response.json().get("response", "").strip()
            if "{" in raw and "}" in raw:
                data = json.loads(raw[raw.index("{"):raw.rindex("}")+1])
                return data.get("action_id", 5), data.get("reply", "已收到")
            return 5, "未能解析指令"
        return 5, f"AI服务异常({response.status_code})"
    except Exception as e:
        return 5, f"连接AI失败：{e}"

# ==========================================
# 8. 执行跳转
# ==========================================
def execute_action(action_id, reply_text):
    print(f"🤖 助手回复：{reply_text}")
    if action_id == 5 or action_id is None:
        return False
    action_info = CAMPUS_ACTIONS.get(action_id)
    if action_info and action_info["url"]:
        print(f"🌐 正在跳转：{action_info['name']} -> {action_info['url']}")
        webbrowser.open(action_info["url"])
        return True
    return False

# ==========================================
# 9. ★ 改动2：错误恢复菜单
# ==========================================
def recovery_menu():
    """Qwen听不懂时，让用户选择或直接补充FAQ"""
    print("\n🤖 抱歉，我没完全理解。您可以：")
    print("  [1] 查课表")
    print("  [2] 申请请假")
    print("  [3] 查成绩")
    print("  [4] 人脸采集")
    print("  [5] 校园官网")
    print("  [6] 我要补充新知识（录入FAQ）")
    print("  [0] 跳过")
    choice = input("请选择（0-6）：").strip()

    if choice == '0':
        return
    if choice in ['1', '2', '3', '4', '5']:
        action_id = int(choice) - 1
        execute_action(action_id, f"好的，正在为您打开{CAMPUS_ACTIONS[action_id]['name']}")
    elif choice == '6':
        q = input("请输入问题：").strip()
        a = input("请输入答案：").strip()
        if q and a:
            add_faq_hot(q, a)
    else:
        print("⚠️ 无效选择")

# ==========================================
# 10. 主循环（★ 带多轮记忆 + 错误恢复）
# ==========================================
if __name__ == "__main__":
    print("=" * 55)
    print("🏫 智慧物院 · 校园语音助手")
    print("支持：查课表 / 请假 / 人脸 / 成绩 / 官网 | 语义问答")
    print("说 '退出' 或 '再见' 即可结束")
    print("=" * 55)

    chat_history = []  # ★ 多轮对话历史

    while True:
        try:
            audio_file = record_audio_vad()
            if not audio_file:
                continue

            user_text = speech_to_text(audio_file)
            if not user_text:
                print("⚠️ 未检测到语音\n")
                continue
            print(f"📝 识别结果：{user_text}")

            if any(w in user_text for w in EXIT_WORDS):
                print("👋 感谢使用，再见！")
                break

            # 第一步：查 FAQ 知识库
            faq_answer = query_faq_vector(user_text)
            if faq_answer:
                print(f"📚 知识库回答：{faq_answer}")
                chat_history.append({"user": user_text, "bot": faq_answer})
                print("\n" + "-" * 35 + "\n")
                continue

            # 第二步：Qwen 意图识别（带历史）
            action_id, reply = ask_campus_ai(user_text, chat_history)
            success = execute_action(action_id, reply)

            # ★ 第三步：失败时进入错误恢复
            if not success:
                recovery_menu()

            chat_history.append({"user": user_text, "bot": reply})
            print("\n" + "-" * 35 + "\n")

        except KeyboardInterrupt:
            print("\n👋 感谢使用，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误：{e}\n")
            continue