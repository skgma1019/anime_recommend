# model_loader.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_all_models():
    """
    모든 데이터 파일에서 콘텐츠 및 행동 기반 모델을 로드하고 반환합니다.
    """
    data = {
        'df': None,
        'cosine_sim': None,
        'indices': None,
        'behavioral_map': {}
    }
    NO_IMAGE_URL = "../images/no_img.png"

    # 1. 콘텐츠 기반 모델 로드 (anime-dataset-2023.csv)
    try:
        df = pd.read_csv('../csv/anime-dataset-2023.csv')
        df.rename(columns={'Name': 'title', 'Synopsis': 'synopsis', 'Genres': 'genres', 'Image URL': 'image_url'}, inplace=True)
        df.dropna(subset=['title', 'synopsis', 'genres',], inplace=True)
        df['image_url'] = df['image_url'].fillna(NO_IMAGE_URL)
        df.reset_index(drop=True, inplace=True)

        df['synopsis'] = df['synopsis'].fillna('')
        df['genres'] = df['genres'].fillna('')

        def clean_text(text):
            return str(text).lower().replace('|', ' ') if pd.notna(text) else ''

        df['soup'] = (df['synopsis'] + ' ' + df['genres']) 
        df['soup'] = df['soup'].apply(clean_text)

        tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=2,)
        tfidf_matrix = tfidf.fit_transform(df['soup'])
        
        data['df'] = df
        data['cosine_sim'] = linear_kernel(tfidf_matrix, tfidf_matrix)
        data['indices'] = pd.Series(df.index, index=df['title']).drop_duplicates()
        print("   ✓ 콘텐츠 기반 모델 생성 완료.")
    
    except Exception as e:
        print(f"❌ 콘텐츠 모델 오류: {e}")
        return None # 모델 로드 실패 시

    # 2. 행동 기반 모델 로드 (recommend_anime_5000.csv)
    try:
        df_rec = pd.read_csv('../csv/recommend_anime_5000.csv')
        
        for index, row in df_rec.iterrows():
            title_1 = row['Anime_1_Title']
            title_2 = row['Anime_2_Title']
            
            if title_1 not in data['behavioral_map']:
                data['behavioral_map'][title_1] = set()
            data['behavioral_map'][title_1].add(title_2)

        for key in data['behavioral_map']:
            data['behavioral_map'][key] = list(data['behavioral_map'][key])
        print("   ✓ 행동 기반 맵 생성 완료.")

    except Exception as e:
        print(f"❌ 행동 모델 오류: {e}")

    return data