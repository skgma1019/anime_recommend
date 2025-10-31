# recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

class RecommenderModel:
    """
    모든 추천 로직과 데이터를 캡슐화하는 클래스
    """
    def __init__(self):
        # 클래스 속성으로 초기화
        self.df = None
        self.cosine_sim = None
        self.indices = None
        self.behavioral_map = {}
        
        # 1. 콘텐츠 기반 모델 로드 (anime-dataset-2023.csv)
        try:
            print("📦 Content-Based 모델 로딩 및 설정...")
            self.df = pd.read_csv('../csv/anime-dataset-2023.csv') # self.df 에 할당
            self.df.rename(columns={'Name': 'title', 'Synopsis': 'synopsis', 'Genres': 'genres',}, inplace=True)
            self.df.dropna(subset=['title', 'synopsis', 'genres'], inplace=True)
            self.df.reset_index(drop=True, inplace=True)

            self.df['synopsis'] = self.df['synopsis'].fillna('')
            self.df['genres'] = self.df['genres'].fillna('')

            def clean_text(text):
                if pd.isna(text):
                    return ''
                return str(text).lower().replace('|', ' ')

            self.df['soup'] = (self.df['synopsis'] + ' ' + self.df['genres']) 
            self.df['soup'] = self.df['soup'].apply(clean_text)

            tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=2,)
            tfidf_matrix = tfidf.fit_transform(self.df['soup'])
            self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) # self.cosine_sim 에 할당
            self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates() # self.indices 에 할당
            print("   ✓ 콘텐츠 기반 모델 생성 완료.")
        
        except Exception as e:
            print(f"❌ 콘텐츠 모델 오류: {e}")
            self.df = None # 실패 시 self.df만 None으로 설정


        # 2. 행동 기반 모델 로드 (recommend_anime_5000.csv)
        try:
            print("📦 Behavioral 모델 로딩 및 설정...")
            df_rec = pd.read_csv('../csv/recommend_anime_5000.csv')
            
            for index, row in df_rec.iterrows():
                title_1 = row['Anime_1_Title']
                title_2 = row['Anime_2_Title']
                
                if title_1 not in self.behavioral_map:
                    self.behavioral_map[title_1] = set()
                self.behavioral_map[title_1].add(title_2)

            for key in self.behavioral_map:
                self.behavioral_map[key] = list(self.behavioral_map[key])
            print("   ✓ 행동 기반 맵 생성 완료.")

        except Exception as e:
            print(f"❌ 행동 모델 오류: {e}")


    def get_hybrid_recommendations(self, title: str, top_n: int = 10):
        """
        주어진 제목에 대해 하이브리드 추천 결과를 반환합니다.
        """
        if self.df is None:
            return None # 모델 로드 실패 시

        final_recommendations = []
        
        # 1. 행동 기반 추천 (새 CSV, 우선순위)
        behavioral_recs = self.behavioral_map.get(title, [])
        for rec_title in behavioral_recs:
            if rec_title != title and rec_title not in final_recommendations:
                final_recommendations.append(rec_title)

        # 2. 콘텐츠 기반 추천 (기존 모델, 보강)
        try:
            idx = self.indices[title]
            sim_scores = sorted(list(enumerate(self.cosine_sim[idx])), key=lambda x: x[1], reverse=True)
            content_recs = self.df['title'].iloc[[i[0] for i in sim_scores[1:]]].tolist()
            
            # 3. 통합 및 중복 제거
            for rec_title in content_recs:
                if rec_title not in final_recommendations:
                    final_recommendations.append(rec_title)

            return final_recommendations[:top_n]

        except KeyError:
            # 콘텐츠 모델에서 제목을 찾지 못했지만, 행동 모델 결과는 있을 수 있음
            if final_recommendations:
                return final_recommendations[:top_n]
            return None # 완전히 제목을 찾지 못함