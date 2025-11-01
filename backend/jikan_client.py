# jikan_client.py

import httpx
import urllib.parse
import asyncio

JIKAN_BASE_URL = "https://api.jikan.moe/v4"

async def fetch_anime_details(title: str):
    """Jikan API에서 애니메이션 제목으로 상세 정보를 가져오는 비동기 함수"""
    encoded_title = urllib.parse.quote(title) 
    
    # 안정적인 요청 설정을 유지합니다.
    async with httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)) as client:
        url = f"{JIKAN_BASE_URL}/anime?q={encoded_title}&limit=1"
        
        try:
            # Jikan API의 Rate Limit을 피하기 위한 딜레이
            # 이 함수를 호출하기 전에 recommender.py에서 딜레이를 주는 것이 더 효율적입니다.
            response = await client.get(url)
            response.raise_for_status() 
            data = response.json().get('data', [])

            if data:
                details = data[0]
                return {
                    "title": details.get('title'),
                    "image_url": details.get('images', {}).get('jpg', {}).get('image_url'),
                    "score": details.get('score'),
                    "mal_id": details.get('mal_id')
                }
        except Exception as e:
            # 오류가 나도 로그만 남기고 None 반환
            print(f"⚠️ Jikan API 호출 실패 for '{title}': {e}")
            pass 
        return None