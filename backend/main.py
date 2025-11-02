# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
#모듈 가져오기
from recommender import RecommenderService
from db import Base, engine, get_db
from user_router import router as user_router 
from anime_router import router as anime_router
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
# try:
#     # RecommenderService 인스턴스 생성
#     recommender_service = RecommenderService()
#     print("🚀 FastAPI 서버가 추천 모델 로딩을 완료했습니다.")
# except Exception as e:
#     print(f"🚨 모델 초기화 실패: {e}. 서버는 실행되지만 API는 작동하지 않습니다.")

#user_router 모듈 연결
app.include_router(
    user_router,
    prefix="/users",  # 👈 2. 이 라우터의 모든 주소 앞에 "/users"를 붙임
    tags=["Users"]    # 👈 3. /docs 페이지에서 "Users" 그룹으로 묶어줌
)
app.include_router(
    anime_router,         # 👈 NEW (anime_router.py 연결)
    prefix="/animes",     # 👈 주소 http://.../animes 로 시작
    tags=["Animes"]       # 👈 API 문서 태그
)
# --- 3. api 엔드포인트 ---

