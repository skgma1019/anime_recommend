# 애니메이션 추천 서비스

## 소개

이 프로젝트는 사용자가 좋아하는 애니메이션을 기반으로 새로운 작품을 추천해주는 풀스택 웹 서비스입니다.  
**콘텐츠 기반 필터링**과 **행동 기반 필터링**을 결합한 하이브리드 추천 알고리즘을 사용하며, Jikan API(MyAnimeList)와 연동해 최신 애니메이션 정보를 제공합니다.

## 주요 기능

- **애니메이션 검색** — 제목 키워드로 애니메이션 검색
- **인기 애니메이션** — 즐겨찾기 수 기준 인기 목록 제공
- **하이브리드 추천** — TF-IDF 코사인 유사도(콘텐츠 기반) + 행동 데이터(협업 필터링) 결합
- **개인화 추천** — 내 즐겨찾기 목록을 기반으로 한 맞춤 추천
- **즐겨찾기 관리** — 애니메이션 추가/삭제 및 목록 조회
- **회원 시스템** — 회원가입, 로그인(JWT), 비밀번호 변경, 회원 탈퇴
- **추천 피드백** — 추천 결과에 대한 사용자 피드백 저장

## 기술 스택

| 구분 | 기술 |
|------|------|
| **Backend** | FastAPI, Python |
| **Frontend** | React 19, Axios |
| **Database** | PostgreSQL (SQLAlchemy ORM) |
| **인증** | JWT (python-jose, passlib/bcrypt) |
| **추천 모델** | scikit-learn (TF-IDF, cosine similarity), pandas |
| **외부 API** | Jikan API (MyAnimeList) |

## 프로젝트 구조

```
anime_recommend/
├── backend/
│   ├── main.py             # FastAPI 앱 진입점 및 CORS 설정
│   ├── recommender.py      # 하이브리드 추천 서비스 레이어
│   ├── model_loader.py     # TF-IDF 모델 및 행동 데이터 로딩
│   ├── anime_router.py     # 애니메이션 관련 API 라우터
│   ├── user_router.py      # 사용자/인증/즐겨찾기 API 라우터
│   ├── auth.py             # JWT 토큰 생성 및 검증
│   ├── crud.py             # DB CRUD 함수
│   ├── db.py               # DB 연결 및 모델 정의
│   ├── schemas.py          # Pydantic 스키마
│   ├── dependencies.py     # 의존성 주입 (추천기 싱글톤)
│   ├── jikan_client.py     # Jikan API 클라이언트
│   └── data/
│       └── anime_combined.csv
├── frontend/
│   └── src/
│       ├── App.js
│       ├── api/anime.js    # API 통신 함수
│       ├── components/     # 공통 컴포넌트
│       └── pages/          # 페이지 컴포넌트
└── csv/
    ├── anime-dataset-2023.csv      # 콘텐츠 기반 모델 데이터
    └── recommend_anime_5000.csv    # 행동 기반 모델 데이터
```

## API 엔드포인트

### 애니메이션
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/animes/` | 전체 목록 (페이지네이션) |
| GET | `/animes/popular` | 인기 애니메이션 |
| GET | `/animes/search?keyword=` | 키워드 검색 |
| GET | `/animes/recommend?title=` | 제목 기반 추천 |
| GET | `/animes/{anime_id}` | 상세 정보 |

### 사용자
| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/users/register` | 회원가입 |
| POST | `/users/login` | 로그인 (JWT 발급) |
| GET | `/users/me` | 내 정보 조회 |
| PUT | `/users/me` | 비밀번호 변경 |
| DELETE | `/users/me` | 회원 탈퇴 |
| GET | `/users/me/favorites` | 즐겨찾기 목록 |
| POST | `/users/me/favorites` | 즐겨찾기 추가 |
| DELETE | `/users/me/favorites/{anime_id}` | 즐겨찾기 삭제 |
| GET | `/users/me/recommendations` | 개인화 추천 |
| POST | `/users/me/feedback` | 추천 피드백 제출 |

## 사용 방법

### 1. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
DB_USERNAME=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=anime_db

SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 2. 백엔드 실행

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# 패키지 설치
pip install fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv \
            python-jose[cryptography] passlib[bcrypt] scikit-learn pandas httpx

# 서버 실행
cd backend
python -m uvicorn main:app --reload
```

백엔드 서버는 `http://localhost:8000` 에서 실행됩니다.  
API 문서는 `http://localhost:8000/docs` 에서 확인할 수 있습니다.

### 3. 프론트엔드 실행

```bash
cd frontend
npm install
npm start
```

프론트엔드는 `http://localhost:3000` 에서 실행됩니다.

### 4. 데이터베이스 준비

PostgreSQL에서 `anime_db` 데이터베이스를 생성한 후 서버를 실행하면 테이블이 자동으로 생성됩니다.

```sql
CREATE DATABASE anime_db;
```

## 추천 알고리즘

1. **콘텐츠 기반 필터링** — 애니메이션의 시놉시스와 장르를 TF-IDF로 벡터화하고 코사인 유사도로 유사 작품을 찾습니다.
2. **행동 기반 필터링** — 사용자들이 함께 시청한 패턴 데이터(`recommend_anime_5000.csv`)를 기반으로 추천합니다.
3. **하이브리드** — 두 방식의 결과를 통합하고 중복을 제거한 뒤 Jikan API로 최신 정보를 보강하여 반환합니다.

## 라이선스

MIT License
