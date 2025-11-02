# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
#모듈 가져오기
from recommender import RecommenderService
from db import Base, engine, get_db
from user_router import router as user_router 
#DB 테이블 생성
Base.metadata.create_all(bind=engine)
# --- 1. FastAPI 앱 인스턴스 생성 및 설정 ---
app = FastAPI() 

# CORS 설정
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 2. 서버 시작 시 서비스 초기화 ---
try:
    # RecommenderService 인스턴스 생성
    recommender_service = RecommenderService()
    print("🚀 FastAPI 서버가 추천 모델 로딩을 완료했습니다.")
except Exception as e:
    print(f"🚨 모델 초기화 실패: {e}. 서버는 실행되지만 API는 작동하지 않습니다.")

#user_router 모듈 연결
app.include_router(
    user_router,
    prefix="/users",  # 👈 2. 이 라우터의 모든 주소 앞에 "/users"를 붙임
    tags=["Users"]    # 👈 3. /docs 페이지에서 "Users" 그룹으로 묶어줌
)

# --- 3. api 엔드포인트 ---

#추천 시스템
@app.get("/recommend")
async def recommend_anime(title: str):
    """
    애니메이션 제목을 받아 하이브리드 추천 결과를 Jikan API 정보와 함께 반환하는 API
    """
    # 서비스 로드 실패 확인
    if not hasattr(recommender_service, 'is_loaded') or not recommender_service.is_loaded: 
        raise HTTPException(status_code=503, detail="서버가 초기화 중이거나 데이터 로딩에 실패했습니다.")

    # 비동기 함수 호출
    recommended_data = await recommender_service.get_enriched_recommendations(title=title)
    
    if recommended_data is None or not recommended_data:
        raise HTTPException(status_code=404, detail=f"'{title}' 제목을 찾을 수 없습니다.")

    return {"recommendations": recommended_data}

#검색 시스템
@app.get("/search")
def search_anime(keyword: str):

    """
    애니메이션 제목 검색 및 자동 완성 기능을 제공하는 API
    """
    if not hasattr(recommender_service, 'is_loaded') or not recommender_service.is_loaded: 
        raise HTTPException(status_code=503, detail="서버가 초기화 중이거나 데이터 로딩에 실패했습니다.")

    # 서비스의 검색 메서드 호출
    matching_titles = recommender_service.search_anime_titles(keyword=keyword)
    
    if not matching_titles:
        raise HTTPException(status_code=404, detail=f"'{keyword}' 키워드로 검색된 제목이 없습니다.")

    return {"titles": matching_titles}

