# crud.py

from sqlalchemy.orm import Session
from db import User  # db.py의 User 모델
from schemas import UserCreate # schemas.py의 UserCreate 모델


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