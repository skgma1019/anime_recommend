# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from recommender import RecommenderModel # <- 이미 수정됨

# --- 1. FastAPI 앱 생성 및 설정 ---
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


# --- 2. 서버 시작 시 모델 초기화 ---
try:
    recommender = RecommenderModel()
    print("🚀 FastAPI 서버가 추천 모델 로딩을 완료했습니다.")
except Exception as e:
    # 모델 로드 실패 시에도 서버는 켜지지만, API는 오류를 반환하도록 합니다.
    # RecommenderModel의 __init__이 오류를 냈다면 recommender 객체 자체가 없을 수 있으므로
    # 이 부분은 그대로 두거나, 더 안전하게 처리합니다.
    print(f"🚨 모델 초기화 실패: {e}. 서버는 실행되지만 API는 작동하지 않습니다.")


# --- 3. API 엔드포인트 정의 ---
@app.get("/recommend")
def recommend_anime(title: str):
    """
    애니메이션 제목을 받아 하이브리드 추천 결과를 반환하는 API
    """
    # 🚨 오류 해결: recommender.df 대신 recommender 객체 자체에 df 속성이 있는지 확인
    # RecommenderModel 내부에 df가 None이면 모델 로드 실패로 간주
    if not hasattr(recommender, 'df') or recommender.df is None: 
        raise HTTPException(status_code=503, detail="서버가 초기화 중이거나 데이터 로딩에 실패했습니다.")

    # RecommenderModel의 메서드를 호출하여 추천 결과를 가져옵니다.
    recommended_titles = recommender.get_hybrid_recommendations(title=title)
    
    if recommended_titles is None:
        raise HTTPException(status_code=404, detail=f"'{title}' 제목을 찾을 수 없습니다.")

    # 결과를 JSON 형식으로 반환
    return {"recommendations": recommended_titles}