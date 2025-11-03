# user_router.py

# 1. 필요한 모든 모듈을 import 합니다.
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
import schemas
import crud
import auth
from auth import get_current_user
from db import get_db, User
from typing import List
from recommender import RecommenderService
from dependencies import get_recommender_service# user_router.py (파일 상단)
import asyncio  # 👈 [추가] 딜레이를 위한 asyncio
from jikan_client import fetch_anime_details # 👈 [추가] Jikan 클라이언트
# 2. 'app = FastAPI()' 대신 'APIRouter()'를 사용합니다.
router = APIRouter()


# 3. '@app.post(...)' 대신 '@router.post(...)'를 사용합니다.
#    주소는 "/register"로 지정합니다. (prefix는 main.py에서 붙여줌)
@router.post("/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    새로운 사용자를 생성 (회원가입)
    - 주소: /users/register
    """
    # 1. 이메일 중복 확인
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # 2. 비밀번호 해시
    hashed_password = auth.get_password_hash(user.password)
    
    # 3. DB에 사용자 생성
    new_user = crud.create_user(db=db, user=user, hashed_password=hashed_password)
    
    # 4. 생성된 사용자 정보 반환
    return new_user

# 4. 로그인 엔드포인트
@router.post("/login", response_model=schemas.Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    사용자 이메일(username)과 비밀번호(password)로 로그인하여 JWT 토큰을 발급받습니다.
    - form_data: OAuth2 표준 폼 데이터 (username, password)
    """
    
    # 1. 이메일(username)로 사용자 확인
    # (OAuth2 폼은 'username' 필드를 사용하므로, 우리 DB의 'email'과 매칭)
    user = crud.get_user_by_email(db, email=form_data.username)
    
    # 2. 사용자가 없거나 비밀번호가 틀리면 에러
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401, # 401 Unauthorized
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # 3. 토큰 만료 시간 설정
    # 3. 토큰 만료 시간 설정
    # 3. 토큰 만료 시간 설정
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # 4. 토큰 생성 (Payload에 사용자 이메일 저장)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    # 5. 토큰 반환
    return {"access_token": access_token, "token_type": "bearer"}

# 5. 본인 정보 엔드포인트
@router.get("/me", response_model=schemas.User)
def read_users_me(
    current_user: schemas.User = Depends(auth.get_current_user)
):  # auth.get_current_user 함수가 성공적으로 사용자를 반환하면,
    # 그 사용자를 'current_user' 변수에 넣어줍니다.
    # 우리는 그냥 그 사용자를 반환하기만 하면 됩니다.
    return current_user

# 6. 비번 수정 엔드포인트
@router.put("/me", response_model=schemas.User)
def update_user_me(
    password_data: schemas.UserUpdatePassword, # Body에서 JSON 받기
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_user) # 토큰 검사
):
    # 1. 새로 받은 비밀번호를 해시
    hashed_password = auth.get_password_hash(password_data.new_password)
    
    # 2. CRUD 함수를 호출하여 DB 업데이트
    updated_user = crud.update_user_password(
        db=db, 
        user=current_user, 
        hashed_password=hashed_password
    )
    
    return updated_user

# 7. 탈퇴 엔드포인트
@router.delete("/me", response_model=schemas.User)
def delete_user_me(
    current_user: schemas.User = Depends(auth.get_current_user), # 토큰 검사
    db: Session = Depends(get_db)
): 
    # 1. CRUD 함수를 호출하여 DB에서 삭제
    #    current_user는 auth.get_current_user가 찾아준 db.User 객체입니다.
    deleted_user = crud.delete_user(db=db, user=current_user)
    
    # 2. 삭제된 사용자 정보 반환
    return deleted_user

# 8. 즐겨찾기 하기
@router.post("/me/favorites", response_model=schemas.UserFavorite)
def create_favorite_for_user(
    favorite: schemas.UserFavoriteCreate, # 1. Body로 anime_id, title 등 받기
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user) # 2. 토큰으로 "나" 확인
):
    # 3. 중복 체크
    db_favorite = crud.get_favorite_by_anime_id(
        db, user_id=current_user.id, anime_id=favorite.anime_id
    )
    if db_favorite:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="이미 즐겨찾기에 추가된 애니메이션입니다."
        )
    
    # 4. CRUD를 통해 DB에 생성
    return crud.create_user_favorite(
        db=db, favorite=favorite, user_id=current_user.id
    )

# 9. 즐겨찾기 목록 조회 엔드포인트
@router.get("/me/favorites", response_model=List[schemas.UserFavorite])
def read_user_favorites(
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user) # 1. 토큰으로 "나" 확인
):
    # 2. CRUD를 통해 DB에서 "내" 목록 조회
    return crud.get_user_favorites(db, user_id=current_user.id)

# 10. 즐겨찾기 삭제 엔드포인트
# user_router.py (기존 함수를 '삭제'하고 '대체')
@router.delete("/me/favorites/{anime_id}", response_model=schemas.UserFavorite)
def delete_favorite_for_user(
    anime_id: int, # 1. URL 경로에서 anime_id 받기
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user) # 2. 토큰으로 "나" 확인
):
    # 3. 삭제할 항목이 DB에 있는지 (내 것이 맞는지) 확인
    db_favorite = crud.get_favorite_by_anime_id(
        db, user_id=current_user.id, anime_id=anime_id
    )
    if db_favorite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="즐겨찾기 목록에 없는 애니메이션입니다."
        )
        
    # 4. CRUD를 통해 DB에서 삭제
    return crud.delete_user_favorite(db, db_favorite=db_favorite)
@router.get("/me/recommendations", response_model=schemas.AnimeListResponse) # 👈 응답 모델 변경
async def get_personalized_recommendations(
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user), 
    recommender: RecommenderService = Depends(get_recommender_service) 
):
    """
    사용자의 즐겨찾기 목록을 기반으로 개인화 추천을 반환합니다. (수정된 버전)
    """
    if not recommender.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="추천 모델이 준비되지 않았습니다.")

    # 3. DB에서 현재 사용자의 찜 목록을 가져옵니다.
    favorites = crud.get_user_favorites(db, user_id=current_user.id)
    if not favorites:
        return {"recommendations": []} # 👈 빈 목록이어도 404 대신 빈 배열 반환
        
    # 4. 찜 목록에서 '제목' 리스트만 추출
    favorite_titles = [fav.title for fav in favorites]
    
    # 5. recommender에게 추천 제목 리스트를 요청합니다. (Jikan 보강 전)
    recommended_titles = recommender.get_personalized_titles(favorite_titles, top_n=20)
    
    if not recommended_titles:
        return {"recommendations": []} # 👈 추천 결과가 없어도 빈 배열 반환

    # 6. [수정됨] Jikan API 정보 보강을 '올바르게' 수행합니다.
    final_recommendations = []
    
    # Jikan API Rate Limit(1초 3회)을 피하기 위해 0.5초 딜레이
    for i, title in enumerate(recommended_titles):
        
        # 🌟 [핵심] 0.5초 딜레이 (첫 번째 요청도 포함!)
        if i > 0: # 0.5초씩 쉬어줍니다.
            await asyncio.sleep(0.5) 
        
        # 🌟 [핵심] 'recommender'가 아니라 'jikan_client'를 직접 호출
        enriched_data = await fetch_anime_details(title)
        
        if enriched_data:
            final_recommendations.append(enriched_data)
            
        # 10개의 유효한 추천을 찾으면 중단
        if len(final_recommendations) >= 10:
            break
            
    if not final_recommendations:
        return {"recommendations": []} # 👈 Jikan 보강 실패 시 빈 배열 반환

    return {"recommendations": final_recommendations}

@router.post("/me/feedback", response_model=schemas.Feedback)
def create_feedback_for_user(
    feedback: schemas.FeedbackCreate, # 👈 1. 요청 Body
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user) # 👈 2. 인증
):
    """
    현재 로그인한 사용자의 추천 피드백을 저장합니다.
    """
    return crud.create_user_feedback(db=db, feedback=feedback, user_id=current_user.id)

# anime_router.py (기존 함수를 삭제하고 이걸로 '대체')

