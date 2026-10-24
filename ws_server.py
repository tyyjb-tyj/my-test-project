import asyncio
import sys
import time
import websockets

# Windows 中文控制台默认 GBK，print 表情符号会抛 UnicodeEncodeError；统一转 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

async def handle_esp32(websocket):
    print(f"✅ ESP32 已连接：{websocket.remote_address}")
    last_log = time.monotonic()
    bytes_since_log = 0
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # 收到音频，直接原样回传（Echo 测试）
                await websocket.send(message)
                bytes_since_log += len(message)
                # 每秒打印一次统计，防止刷屏
                now = time.monotonic()
                if now - last_log >= 1.0:
                    print(f"🎵 回传中：近 1 秒 {bytes_since_log} 字节")
                    bytes_since_log = 0
                    last_log = now
            else:
                print(f"📩 收到文本：{message}")
    except websockets.exceptions.ConnectionClosed:
        print("❌ ESP32 已断开")

async def main():
    # 监听所有网卡，防止只在 localhost 监听导致 ESP32 连不上
    async with websockets.serve(handle_esp32, "0.0.0.0", 12345, max_size=10*1024*1024):
        print("🚀 Echo 服务器已启动，监听 0.0.0.0:12345")
        print("   ⚠️ 如果弹出防火墙提示，请务必勾选『允许访问』！")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
