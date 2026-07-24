import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np

load_dotenv()   # ← .env 로드

emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

sents = [
    "어젯밤 늦게까지 잠을 못 잤다",
    "밤이 깊어 사방이 조용하다",
    "가을에 밤을 주워 구워 먹었다",
    "밤나무에서 밤송이가 떨어졌다",
]

vecs = [np.array(emb.embed_query(s)) for s in sents]

def cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"시간-시간: {cos(vecs[0], vecs[1]):.4f}")
print(f"음식-음식: {cos(vecs[2], vecs[3]):.4f}")
print(f"시간-음식: {cos(vecs[0], vecs[2]):.4f}")