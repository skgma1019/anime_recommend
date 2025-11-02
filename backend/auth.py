import os
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import crud
from db import get_db
# auth.py

# 1. 암호화 방식 설정 (bcrypt 사용)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="users/login")

# 2. 비밀번호 검증 함수 (로그인 시 사용)
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """입력된 비밀번호와 DB의 해시된 비밀번호를 비교합니다."""
    return pwd_context.verify(plain_password, hashed_password)

# 3. 비밀번호 해시(암호화) 함수 (회원가입 시 사용)
def get_password_hash(password: str) -> str:
    """입력된 비밀번호를 bcrypt 해시로 변환합니다."""
    if len(password.encode("utf-8")) > 72:
        password = password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
    return pwd_context.hash(password)

# 4. JWT 토큰 생성 함수
def create_access_token(data: dict, expires_delta: timedelta | None = None):

    """
    주어진 데이터(data)와 만료 시간(expires_delta)으로 JWT 토큰을 생성합니다.
    """
    to_encode = data.copy()
    
    # 만료 시간 설정
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # 기본 만료 시간 (15분)
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
        
    to_encode.update({"exp": expire})
    
    # JWT 토큰 생성 (서명)
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt

# 5. 토큰 검사 및 사용자 반환
def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: Session = Depends(get_db)
):
    """
    1. 요청 헤더에서 Bearer 토큰을 가져옵니다.
    2. 토큰을 검증(decode)합니다.
    3. 토큰의 'sub' (이메일) 값으로 DB에서 사용자를 찾아 반환합니다.
    """
    
    # 토큰 검증 실패 시 사용할 예외
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # 1. 토큰 검증 (서명, 만료 시간 등)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # 2. 토큰에서 이메일(sub) 추출
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
            
    except JWTError:
        # 3. 토큰 디코딩 실패 시 (위조, 만료 등)
        raise credentials_exception
        
    # 4. 이메일로 DB에서 사용자 조회
    user = crud.get_user_by_email(db, email=email)
    
    if user is None:
        # 5. 토큰은 유효하지만, 그 사이 사용자가 DB에서 삭제된 경우
        raise credentials_exception
        
    # 6. 검문 통과! (사용자 객체 반환)
    return user
