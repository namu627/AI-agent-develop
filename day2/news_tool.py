import requests
import os
from dotenv import load_dotenv
load_dotenv()

def search_news(keyword: str) -> str:
    """키워드로 뉴스를 검색하는 Tool"""
    url= "https://newsdata.io/api/1/news"
    params = {
        "apikey": os.getenv("NEWSDATA_API_KEY"),
        "q": keyword,
        "language": "ko",
        "size": 5
        }
    response = requests.get(url, params=params)
    articles = response.json().get("results", [])
    result = ""
    for article in articles:
        result += f"제목: {article['title']}\n"
        result += f"내용: {article.get('description', '내용 없음')}\n\n"
        return result if result else "관련 뉴스를 찾을 수 없습니다."
# 함수 실행 테스트
print(search_news("인공지능"))