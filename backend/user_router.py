# user_router.py

# 1. 필요한 모든 모듈을 import 합니다.
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
import schemas
import crud
import auth
from db import get_db

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