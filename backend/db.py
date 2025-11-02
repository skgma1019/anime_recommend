# db.py
import os  # 1. os 라이브러리 가져오기
from dotenv import load_dotenv  # 2. dotenv 라이브러리 가져오기
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship

# -------------------------------------------------
# 1. DB 접속 설정 (PostgreSQL)
# -------------------------------------------------
# "postgresql://유저명:비밀번호@localhost:5432/DB이름"
# 아래 'postgres'와 'anime_db'는 pgAdmin에서 설정한 값입니다.

load_dotenv()
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
# 최종 접속 URL 생성
SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# DB 엔진 생성
engine = create_engine(SQLALCHEMY_DATABASE_URL)


# -------------------------------------------------
# 2. DB 세션 생성 및 관리
# -------------------------------------------------
# DB와 통신(읽기, 쓰기)할 때 사용할 '세션(Session)'을 만듭니다.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 모든 DB 모델(테이블)이 상속받을 'Base' 클래스입니다.
Base = declarative_base()


# API 엔드포인트에서 DB 세션을 쉽게 사용하고 닫을 수 있게 도와주는 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------------------------------
# 3. User 모델(테이블) 정의
# -------------------------------------------------
# 'users'라는 이름의 테이블 설계도입니다.
class User(Base):
    __tablename__ = "users"

    # id: 유저 고유 번호 (자동 증가)
    id = Column(Integer, primary_key=True, index=True)
    
    # email: 유저 아이디로 사용할 이메일 (중복 불가)
    email = Column(String, unique=True, index=True, nullable=False)
    
    # hashed_password: 암호화되어 저장될 비밀번호
    hashed_password = Column(String, nullable=False)
    
    # is_active: 활성 계정 여부 (예: 탈퇴 처리)
    is_active = Column(Boolean, default=True)
    
    favorites = relationship("UserFavorite", back_populates="owner")

class UserFavorite(Base):
    __tablename__ = "user_favorites"

    id = Column(Integer, primary_key=True, index=True)
    
    # 1. "누가" 찜했는지 (users 테이블의 id와 연결)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # 2. "무엇을" 찜했는지 (recommender.df의 anime_id)
    anime_id = Column(Integer, index=True)
    
    # 3. 목록을 보여줄 때 사용할 정보 (DB에 따로 저장)
    title = Column(String)
    image_url = Column(String, nullable=True)

    # User 모델과 UserFavorite 모델을 연결 (선택 사항이지만 권장됨)
    owner = relationship("User", back_populates="favorites")

