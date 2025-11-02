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