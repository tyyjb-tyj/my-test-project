import pymysql

# 连接数据库
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='123456',
    database='hmlc_college',
    charset='utf8mb4'
)

cursor = conn.cursor()

# 查询 FAQ
cursor.execute("SELECT question, answer FROM faq LIMIT 5")
results = cursor.fetchall()

for q, a in results:
    print(f"Q: {q}")
    print(f"A: {a}")
    print("-" * 40)

conn.close()