# anime_router.py (수정본)

from fastapi import APIRouter, Depends, HTTPException
from typing import List
import schemas # 👈 1. schemas import (List[schemas.Anime] 때문)

# 👈 2. recommender 모듈과 의존성 함수 import
from recommender import RecommenderService
from dependencies import get_recommender_service
import crud
from sqlalchemy.orm import Session
from db import get_db

# 3. '@app' 대신 '@router'를 사용하기 위해 APIRouter 생성
router = APIRouter()


# ---------------------------------------------
# 1. 모든 애니 목록 (페이지네이션)
# ---------------------------------------------
@router.get("/", response_model=List[schemas.Anime])
def read_animes(
    skip: int = 0,
    limit: int = 20,
    # 'Depends'를 통해 초기화된 추천기 객체를 받음
    recommender: RecommenderService = Depends(get_recommender_service),
    db: Session = Depends(get_db)
):
    if not recommender.is_loaded:
        raise HTTPException(status_code=503, detail="모델이 아직 로드 중입니다.")
    animes_list = recommender.get_all_animes(skip=skip, limit=limit)
    for anime_dict in animes_list:
        anime_id = anime_dict.get("anime_id") # 각 애니의 ID를 얻음
        
        # crud.py의 함수를 호출하여 DB에서 카운트 조회
        count = crud.get_favorites_count_by_anime_id(db, anime_id=anime_id)
        
        # 딕셔너리에 'favorites_count' 키로 카운트 값 추가
        anime_dict["favorites_count"] = count
    return animes_list


# ---------------------------------------------
# 2. 추천 시스템 (Jikan API 연동)
# ---------------------------------------------
# 🌟 (오류 수정: @app -> @router)
@router.get("/recommend") 
async def recommend_anime(
    title: str,
    # 🌟 (오류 수정: 'recommender_service' -> 'recommender' 객체 받기)
    recommender: RecommenderService = Depends(get_recommender_service)
):
    if not recommender.is_loaded: 
        raise HTTPException(status_code=503, detail="서버가 초기화 중이거나 데이터 로딩에 실패했습니다.")

    # 🌟 (오류 수정: 'recommender_service' -> 'recommender' 사용)
    recommended_data = await recommender.get_enriched_recommendations(title=title)
    
    if recommended_data is None or not recommended_data:
        raise HTTPException(status_code=404, detail=f"'{title}' 제목을 찾을 수 없습니다.")

    # (팁: schemas.py에 응답 모델을 정의하면 더 좋습니다)
    return {"recommendations": recommended_data}


# ---------------------------------------------
# 3. 검색 시스템
# ---------------------------------------------
# 🌟 (오류 수정: @app -> @router)
@router.get("/search")
def search_anime(
    keyword: str,
    # 🌟 (오류 수정: 'recommender_service' -> 'recommender' 객체 받기)
    recommender: RecommenderService = Depends(get_recommender_service)
):
    if not recommender.is_loaded: 
        raise HTTPException(status_code=503, detail="서버가 초기화 중이거나 데이터 로딩에 실패했습니다.")

    # 🌟 (오류 수정: 'recommender_service' -> 'recommender' 사용)
    matching_titles = recommender.search_anime_titles(keyword=keyword)
    
    if not matching_titles:
        raise HTTPException(status_code=404, detail=f"'{keyword}' 키워드로 검색된 제목이 없습니다.")

    # (팁: schemas.py에 응답 모델을 정의하면 더 좋습니다)
    return {"titles": matching_titles}