#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


# In[2]:


# TMDB_API_KEY = '287e229999edb0afe4bf0df40ad08ce7'

BASE_URL = 'https://api.jikan.moe/v4'


# In[3]:


import requests

BASE_URL = "https://api.jikan.moe/v4"

def fetch_top_anime(page=1):
    """Jikan API에서 인기 애니메이션 목록 가져오기"""
    url = f"{BASE_URL}/top/anime"
    params = {
        'page': page
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get('data', [])  # 'results'가 아니라 'data'입니다!
    except Exception as e:
        print(f"⚠️ 인기 애니메이션 가져오기 실패: {e}")
        return []


def fetch_anime_details(mal_id):
    """Jikan API에서 개별 애니메이션 상세 정보 가져오기"""
    url = f"{BASE_URL}/anime/{mal_id}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get('data', {})  # ✅ 'data' 키로 접근
    except Exception as e:
        print(f"⚠️ 애니메이션 상세 정보 실패 (ID: {mal_id}): {e}")
        return {}


# In[4]:


def fetch_top_rated_anime(page=1):
    """높은 평점 애니 가져오기"""
    url = f"{BASE_URL}/top/anime"
    params = {
        'filter': 'bypopularity',  # 인기도순 (혹은 'favorite', 'airing' 등으로 변경 가능)
        'page': page
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        print(f"⚠️ 높은 평점 애니 가져오기 실패: {e}")
        return []

def fetch_now_airing_anime(page=1):
    """현재 방영 중인 애니 가져오기"""
    url = f"{BASE_URL}/seasons/now"
    params = {
        'page': page
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        print(f"⚠️ 방영 애니 가져오기 실패: {e}")
        return []


# In[5]:


import time

def collect_anime_by_category(fetch_func, category_name, num_anime=500):
    print(f"📦 {category_name} 애니메이션 수집 중...")

    anime_list = []
    anime_ids = set()
    page = 1

    while len(anime_list) < num_anime:
        result = fetch_func(page)

        if not result:
            print(f"⚠️ 더 이상 데이터 없음 (총 {len(anime_list)}개 수집)")
            break

        for anime in result:
            if anime['mal_id'] not in anime_ids and len(anime_list) < num_anime:
                anime_ids.add(anime['mal_id'])
                anime_list.append(anime)

        print(f"\r진행: {len(anime_list)}/{num_anime}...", end='\r')

        page += 1
        time.sleep(1.5)  # Jikan API rate limit 준수

    print(f"\n✅ {category_name}: 총 {len(anime_list)}개 수집 완료")
    return anime_list


# In[6]:


def collect_anime_data(num_per_category=500):
    print(f"📥 Jikan API에서 애니메이션 데이터 수집 중 (각 카테고리 {num_per_category}개)")
    print("=" * 80)

    all_anime = []
    anime_ids = set()

    # 1. 인기 애니 수집
    popular_anime = collect_anime_by_category(
        fetch_top_rated_anime,
        'Popular',
        num_per_category
    )

    # 2. 높은 평점 애니 수집
    top_rated_anime = collect_anime_by_category(
        fetch_top_rated_anime,
        'Top Rated',
        num_per_category
    )

    # 3. 현재 방영 애니 수집
    now_airing_anime = collect_anime_by_category(
        fetch_now_airing_anime,
        'Now Airing',
        num_per_category
    )

    # 4. 병합 (중복 제거)
    print("\n🔄 데이터 병합 중 (중복 제거)...")
    for anime in popular_anime + top_rated_anime + now_airing_anime:
        if anime['mal_id'] not in anime_ids:
            anime_ids.add(anime['mal_id'])
            all_anime.append(anime)

    print(f"  ✓ 총 수집: {len(popular_anime) + len(top_rated_anime) + len(now_airing_anime)}개")
    print(f"  ✓ 중복 제거 후: {len(all_anime)}개")

    print("\n" + "=" * 80)
    print(f"✅ 기본 정보 수집 완료: {len(all_anime)}개")

    # 상세 정보 수집
    print("\n📝 애니메이션 상세 정보 수집 중...")
    print("  (시간이 걸릴 수 있습니다...)")
    detailed_anime = []

    total = len(all_anime)
    for i, anime in enumerate(all_anime, 1):
        if i % 50 == 0 or i == total:
            print(f"  진행: {i}/{total} ({i/total*100:.1f}%)...")

        details = fetch_anime_details(anime['mal_id'])  # <- 여기가 핵심
        if details:
            detailed_anime.append(details)

        time.sleep(0.5)  # Jikan API 요청 제한

    print(f"\n✅ 상세 정보 수집 완료: {len(detailed_anime)}개")

    # 카테고리별 통계
    print("\n📊 카테고리별 통계:")
    print(f"  - Popular: {len(popular_anime)}개")
    print(f"  - Top Rated: {len(top_rated_anime)}개")
    print(f"  - Now Airing: {len(now_airing_anime)}개")
    print(f"  - 최종 (중복 제거): {len(detailed_anime)}개")

    return detailed_anime


# In[7]:


def parse_anime_data(anime_data_list):
    """
    Jikan API 응답 데이터를 DataFrame으로 변환
    """
    print("\n🔧 애니메이션 데이터 파싱 중...")

    parsed_anime = []

    for anime in anime_data_list:
        genres = '|'.join([g['name'] for g in anime.get('genres', [])])
        studios = '|'.join([s['name'] for s in anime.get('studios', [])])
        themes = '|'.join([t['name'] for t in anime.get('themes', [])])
        demographics = '|'.join([d['name'] for d in anime.get('demographics', [])])

        parsed = {
            'mal_id': anime['mal_id'],
            'title': anime.get('title', ''),
            'title_english': anime.get('title_english', ''),
            'synopsis': anime.get('synopsis', ''),
            'genres': genres,
            'themes': themes,
            'score': anime.get('score', 0),
            'scored_by': anime.get('scored_by', 0),
            'popularity': anime.get('popularity', 0),
            'rank': anime.get('rank', 0),
            'season': anime.get('season', ''),
            'year': anime.get('year', ''),
          }

        parsed_anime.append(parsed)

    df = pd.DataFrame(parsed_anime)
    print(f"  ✓ 파싱 완료: {len(df)}개 애니메이션")
    return df


# In[8]:


print("데이터 수집 시작...")
print()

# 1. 애니메이션 데이터 수집 (Top, Now, 등 다양한 카테고리 기반)
anime_raw = collect_anime_data(num_per_category=100)

# 2. 상세 정보를 포함한 데이터를 파싱하여 DataFrame으로 변환
df = parse_anime_data(anime_raw)

# 3. 요약 통계 출력
print("\n" + "=" * 80)
print("📊 수집된 애니메이션 데이터 정보")
print("=" * 80)
print(f"총 애니 수: {len(df)}")
print(f"평균 인기도: {df['popularity'].mean():.1f}")
print(f"\n상위 5개 애니:")
print(df[['title', 'season', 'year']].head())



# In[9]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# =============================
# 1. 결측치 처리 및 텍스트 클린
# =============================

df['synopsis'] = df['synopsis'].fillna('')
df['genres'] = df['genres'].fillna('')
df['themes'] = df['themes'].fillna('')


# 텍스트 정제 함수
def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = text.replace('|', ' ')
    return text

# soup 컬럼 생성 (텍스트 통합)
df['soup'] = (
    df['synopsis'] + ' ' +
    df['genres'] + ' ' +
    df['themes']
)

df['soup'] = df['soup'].apply(clean_text)

print("✅ 애니메이션 텍스트 통합 및 정제 완료")

# =============================
# 2. TF-IDF 벡터화 및 유사도 계산
# =============================

print("\n🔢 TF-IDF 벡터화 중...")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
)

tfidf_matrix = tfidf.fit_transform(df['soup'])
print(f"✓ TF-IDF 행렬 크기: {tfidf_matrix.shape}")

print("\n🔍 코사인 유사도 계산 중...")
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(f"✓ 유사도 행렬 크기: {cosine_sim.shape}")

# =============================
# 3. 인덱스 매핑
# =============================

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# =============================
# 4. 추천 함수
# =============================

def get_recommendations(title, cosine_sim=cosine_sim, df=df, top_n=10):
    """애니메이션 추천 함수"""
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]

        anime_indices = [i[0] for i in sim_scores]

        recommendations = df.iloc[anime_indices][['title', 'genres', 'score', 'popularity']].copy()
        recommendations['similarity_score'] = [score[1] for score in sim_scores]

        return recommendations

    except KeyError:
        print(f"\n❌ '{title}' 애니메이션을 찾을 수 없습니다.")
        print(f"\n사용 가능한 애니 제목 예시:")
        for i, anime in enumerate(df['title'].head(10), 1):
            print(f"  {i}. {anime}")
        return None

# =============================
# 5. 검색 함수
# =============================

def search_anime(keyword, df=df, top_n=10):
    """애니 제목 검색"""
    results = df[df['title'].str.contains(keyword, case=False, na=False)]
    if len(results) == 0:
        print(f"\n'{keyword}'로 검색된 애니가 없습니다.")
        return None
    return results[['title', 'genres', 'score']].head(top_n)


# =============================
# 6. 사용 예시
# =============================

# 🔍 제목 검색
# search_anime("Naruto")

# 🎯 추천 받기
get_recommendations("One Punch Man")


# In[10]:


print("\n" + "=" * 80)
print("🎯 대화형 애니메이션 추천 시스템")
print("=" * 80)

def interactive_recommendation():
    """대화형 애니 추천 시스템"""
    print("\n사용 가능한 애니 목록 (랜덤 20개):")

    # 랜덤하게 20개 선택
    sample_size = min(20, len(df))
    random_animes = df.sample(sample_size)['title'].tolist()

    for i, anime in enumerate(random_animes, 1):
        print(f"  {i}. {anime}")

    print("\n" + "-" * 80)
    print("명령어:")
    print("  - 애니 제목 입력: 해당 애니와 유사한 애니 추천")
    print("  - 'search [키워드]': 애니 제목 검색")
    print("  - 'list': 전체 애니 목록 보기")
    print("  - 'random': 랜덤 20개 다시 보기")
    print("  - 'quit': 종료")
    print("-" * 80)

    while True:
        user_input = input("\n🎬 애니 제목을 입력하세요: ").strip()

        if user_input.lower() == 'quit':
            print("\n👋 추천 시스템을 종료합니다!")
            break

        elif user_input.lower() == 'random':
            print("\n🎲 랜덤 애니 20개:")
            sample_size = min(20, len(df))
            random_animes = df.sample(sample_size)['title'].tolist()
            for i, anime in enumerate(random_animes, 1):
                print(f"  {i}. {anime}")

        elif user_input.lower() == 'list':
            print(f"\n전체 애니 목록 ({len(df)}편):")
            for i, anime in enumerate(df['title'], 1):
                print(f"  {i}. {anime}")

        elif user_input.lower().startswith('search '):
            keyword = user_input[7:]
            results = search_anime(keyword)
            if results is not None:
                print(f"\n🔍 '{keyword}' 검색 결과:")
                print(results.to_string(index=False))

        elif user_input in df['title'].values:
            recommendations = get_recommendations(user_input, top_n=5)
            if recommendations is not None:
                print(f"\n✨ '{user_input}'와 유사한 애니 추천:")
                print("\n" + recommendations.to_string(index=False))

        else:
            print(f"\n❌ '{user_input}' 애니를 찾을 수 없습니다.")
            print("'list' 명령으로 전체 목록을 확인하거나, 'search [키워드]'로 검색하세요.")

# 실행
interactive_recommendation()


# In[ ]:




