# crud.py

from sqlalchemy.orm import Session
from sqlalchemy import func
from db import User  # db.py의 User 모델
from schemas import UserCreate # schemas.py의 UserCreate 모델
from db import User, UserFavorite
from schemas import UserFavoriteCreate

#이메일로 사용자가 있는지 확인
def get_user_by_email(db: Session, email: str):
    
    return db.query(User).filter(User.email == email).first()

# 새로운 사용자 생성
def create_user(db: Session, user: UserCreate, hashed_password: str):
    # 1. DB 모델 객체 생성
    db_user = User(
        email=user.email, 
        hashed_password=hashed_password
    )
    
    # 2. DB 세션에 추가
    db.add(db_user)
    
    # 3. DB에 커밋 (실제 저장)
    db.commit()
    
    # 4. 생성된 객체를 다시 읽어옴 (ID 등 최신 정보 포함)
    db.refresh(db_user)
    
    return db_user

# 사용자 비번 업데이트
def update_user_password(db: Session, user: User, hashed_password: str):

    user.hashed_password = hashed_password
    db.commit()
    db.refresh(user)
    return user

# 회원탈퇴
def delete_user(db: Session, user: User):

    # 1. auth.get_current_user가 찾아준 user 객체를 삭제 대상으로 지정
    db.delete(user)
    
    # 2. DB에 변경 사항(삭제) 저장
    db.commit()
    
    # 3. 삭제된 user 객체 반환 (JSON 응답용)
    return user

#(중복 체크용) 특정 유저가 특정 애니를 찜했는지 확인
def get_favorite_by_anime_id(db: Session, user_id: int, anime_id: int):
    return db.query(UserFavorite).filter(
        UserFavorite.user_id == user_id, 
        UserFavorite.anime_id == anime_id
    ).first()

#특정 유저의 '모든' 즐겨찾기 목록을 조회
def get_user_favorites(db: Session, user_id: int):
    return db.query(UserFavorite).filter(UserFavorite.user_id == user_id).all()

#새로운 즐겨찾기를 DB에 생성
def create_user_favorite(db: Session, favorite: UserFavoriteCreate, user_id: int):
    # 1. 스키마(favorite)와 user_id를 합쳐서 DB 모델 객체 생성
    db_favorite = UserFavorite(
        **favorite.model_dump(),  # anime_id, title, image_url 포함
        user_id=user_id
    )
    
    # 2. DB에 추가 및 저장
    db.add(db_favorite)
    db.commit()
    db.refresh(db_favorite)
    
    return db_favorite

#DB에서 즐겨찾기 레코드 삭제
def delete_user_favorite(db: Session, db_favorite: UserFavorite):
    
    db.delete(db_favorite)
    db.commit()
    return db_favorite

def get_favorites_count_by_anime_id(db: Session, anime_id: int):
    """
    특정 anime_id가 user_favorites 테이블에 몇 번 등장하는지 카운트합니다.
    """
    # SQL: SELECT count(*) FROM user_favorites WHERE anime_id = :anime_id
    count = db.query(func.count(UserFavorite.id)).filter(
        UserFavorite.anime_id == anime_id
    ).scalar() # scalar()는 숫자 값(count)만 반환
    
    return count