# recommender.py

from model_loader import load_all_models
from jikan_client import fetch_anime_details
from fastapi import HTTPException
import asyncio

class RecommenderService:
    """
    추천 모델을 관리하고 API 로직을 실행하는 서비스 레이어
    """
    def __init__(self):
        print("🚀 Recommender Service 초기화...")
        model_data = load_all_models()
        
        if model_data is None:
            self.is_loaded = False
            self.df = None
            return

        # 모델 로드 성공 시, 모든 데이터를 클래스 속성으로 저장
        self.df = model_data['df']
        self.cosine_sim = model_data['cosine_sim']
        self.indices = model_data['indices']
        self.behavioral_map = model_data['behavioral_map']
        self.is_loaded = True


    def search_anime_titles(self, keyword: str, top_n: int = 10):
        """
        데이터베이스에서 키워드를 포함하는 애니메이션 제목을 검색합니다.
        """
        if self.df is None:
            return []

        results = self.df[
            self.df['title'].str.contains(keyword, case=False, na=False)
        ]

        if not results.empty:
            # CSV에 score 컬럼이 있다고 가정하고 점수순으로 정렬
            if 'score' in results.columns:
                 return results.sort_values(by='score', ascending=False)['title'].head(top_n).tolist()
            else:
                 return results['title'].head(top_n).tolist()
        
        return []


    def get_hybrid_recommendations(self, title: str, top_n: int = 20):
        """
        주어진 제목에 대해 하이브리드 추천 결과 (제목 리스트)를 반환합니다.
        """
        if self.df is None:
            return None

        final_recommendations = []
        
        # 1. 행동 기반 추천
        behavioral_recs = self.behavioral_map.get(title, [])
        for rec_title in behavioral_recs:
            if rec_title != title and rec_title not in final_recommendations:
                final_recommendations.append(rec_title)

        # 2. 콘텐츠 기반 추천
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
            if final_recommendations:
                return final_recommendations[:top_n]
            return None


    async def get_enriched_recommendations(self, title: str, top_n: int = 10):
        """
        하이브리드 추천 목록을 만든 후, Jikan API로 최신 정보를 보강하여 반환 (안정화된 버전)
        """
        candidate_titles = self.get_hybrid_recommendations(title, top_n=20) 

        if candidate_titles is None:
            return None

        final_list = []
        
        for i, rec_title in enumerate(candidate_titles):
            # Jikan API Rate Limit을 피하기 위한 딜레이 (0.5초)
            if i > 0:
                 await asyncio.sleep(0.5) 
            
            # jikan_client.py의 비동기 함수 호출
            enriched_data = await fetch_anime_details(rec_title)
            
            if enriched_data and len(final_list) < top_n:
                final_list.append(enriched_data)
                
            if len(final_list) >= top_n:
                break
            
        print(f"✅ Jikan API 정보 보강 완료. 최종 {len(final_list)}개 반환.")
        return final_list
    
    # recommender.py (RecommenderService 클래스 내부에 추가)

    # ... (get_enriched_recommendations 함수 아래에 추가) ...

    def get_all_animes(self, skip: int = 0, limit: int = 20):
        """
        전체 애니메이션 목록을 skip, limit을 이용해 잘라서 반환합니다.
        (self.df를 사용)
        """
        if self.df is None:
            return []
        
        # .iloc[start:end] - DataFrame을 정수 위치로 자릅니다.
        paginated_df = self.df.iloc[skip : skip + limit]
        
        # DataFrame을 Python 딕셔너리 리스트로 변환하여 반환
        return paginated_df.to_dict('records')