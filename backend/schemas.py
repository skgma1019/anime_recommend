# schemas.py

from pydantic import BaseModel, EmailStr

# --- 1. User 생성을 위한 입력 스키마 ---
# (API로 '받을' 데이터 형태)
class UserCreate(BaseModel):
    email: EmailStr  # Pydantic이 이메일 형식을 검증해 줍니다.
    password: str

# --- 2. User 정보를 반환하기 위한 출력 스키마 ---
# (API가 '보낼' 데이터 형태)
# 🚨 절대 비밀번호는 포함하지 않습니다.
class User(BaseModel):
    id: int
    email: EmailStr
    is_active: bool

    # 이 모델이 SQLAlchemy 객체(DB 데이터)를 읽을 수 있게 함
    class Config:
        from_attributes = True

# --- 3. Token 스키마 ---
class Token(BaseModel):
    access_token: str
    token_type: str

# --- 4. 데이터 수정 스키마 ---
class UserUpdatePassword(BaseModel):
    new_password: str

class Anime(BaseModel):
    anime_id: int
    title: str
    genres: str | None = None
    image_url: str | None = None
    score: float| None = None

    favorites_count: int = 0
    
    class Config:
        orm_mode = True

# 즐겨찾기를 '생성'할 때 Body로 받을 정보 (POST /users/me/favorites)
class UserFavoriteCreate(BaseModel):
    anime_id: int
    title: str
    image_url: str | None = None

# 즐겨찾기 정보를 '응답'할 때 사용할 기본 모델
class UserFavorite(UserFavoriteCreate):
    id: int       # DB에서 생성된 고유 ID
    user_id: int  # 누구의 즐겨찾기인지

    class Config:
        orm_mode = True

class FeedbackBase(BaseModel):
    recommendation_type: str # 어떤 종류의 추천인지 (예: 'personal', 'title_based')
    is_satisfied: bool       # 만족 여부 (True/False)
    feedback_text: str | None = None # (선택) 사용자 코멘트

class FeedbackCreate(FeedbackBase):
    pass

class Feedback(FeedbackBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True