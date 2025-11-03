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
    recommender: RecommenderService = Depends(get_recommender_service)
):
    if not recommender.is_loaded: 
        raise HTTPException(status_code=503, detail="서버가 초기화 중이거나 데이터 로딩에 실패했습니다.")

    # 🌟 [수정 1] 'search_anime_titles' -> 'search_anime_objects'로 변경
    matching_animes: list[dict] = recommender.search_anime_objects(keyword=keyword)
    
    if not matching_animes:
        # (검색 결과가 없는 것은 404 에러가 아니라, 빈 배열을 반환하는 것이 더 좋습니다)
        return [] # 빈 배열 반환

    # 🌟 [수정 2] {"titles": ...} 객체가 아닌, '애니메이션 객체 배열' 자체를 반환
    return matching_animes

# anime_router.py (기존 함수를 삭제하고 이걸로 '대체')

@router.get("/popular", response_model=List[schemas.Anime])
def get_popular_animes(
    limit: int = 20,
    db: Session = Depends(get_db), # 👈 'db'는 찜 횟수 계산을 위해 남겨둡니다.
    recommender: RecommenderService = Depends(get_recommender_service)
):
    """
    [수정됨] CSV(df)의 'favorites' 수(Jikan 찜 수)를 기준으로 인기 목록을 반환합니다.
    """
    if not recommender.is_loaded:
        raise HTTPException(status_code=503, detail="모델이 아직 로드 중입니다.")
        
    # 1. [핵심] recommender의 메인 df에 'favorites' 컬럼이 있는지 확인
    # (model_loader.py에서 이 컬럼을 로드해야 합니다)
    if 'favorites' in recommender.df.columns:
        # 'favorites' 컬럼(Jikan 찜 수)을 기준으로 내림차순 정렬
        popular_df = recommender.df.sort_values(by='favorites', ascending=False).head(limit)
    else:
        # 'favorites' 컬럼이 로드되지 않았다면, 'score' (점수)로 대체
        print("⚠️ 'favorites' 컬럼이 df에 없어 'score' 기준으로 대체합니다.")
        if 'score' in recommender.df.columns:
            popular_df = recommender.df.sort_values(by='score', ascending=False).head(limit)
        else:
            # 둘 다 없으면 그냥 20개 반환 (데이터 로드 순)
            popular_df = recommender.df.head(limit)

    # 2. DataFrame을 딕셔너리 리스트로 변환
    results = popular_df.to_dict('records')
    
    # 3. [개선] '우리 DB'의 찜 횟수('favorites_count')도 함께 조회하여 반환
    # (Jikan API의 찜 수와 우리 서비스의 찜 수를 둘 다 보여줄 수 있음)
    for anime_dict in results:
        anime_id = anime_dict.get("anime_id")
        
        # crud.py를 호출하여 '우리 DB'의 찜 횟수 조회
        count = crud.get_favorites_count_by_anime_id(db, anime_id=anime_id)
        
        # 'favorites_count'는 우리 DB 찜 횟수로 설정
        anime_dict["favorites_count"] = count 
        
        # (참고: anime_dict['favorites']에는 Jikan API 찜 수가 이미 들어있음)

    return results

# 상세 정보 엔드포인트
@router.get("/{anime_id}", response_model=schemas.Anime)
def read_anime_details(
    anime_id: int,
    recommender: RecommenderService = Depends(get_recommender_service),
    db: Session = Depends(get_db)
):
    if not recommender.is_loaded:
        raise HTTPException(status_code=503, detail="모델이 아직 로드 중 입니다.")
    
    anime_data = recommender.get_anime_details_by_id(anime_id)
    if anime_data is None:
        raise HTTPException(status_code=404, detail=f"Anime with ID {anime_id} not found.")

    # 찜한 사람 수 계산 로직
    favorite_count = crud.get_favorites_count_by_anime_id(db, anime_id=anime_id)
    anime_data['favorites_count'] = favorite_count

    # 2. 딕셔너리 형태의 데이터를 schemas.Anime 응답 모델로 반환합니다.
    return anime_data

