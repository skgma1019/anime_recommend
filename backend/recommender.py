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
    
    def get_personalized_titles(self, anime_titles: list[str], top_n: int = 10) -> list[str]:
        """
        사용자가 찜한 여러 애니메이션 제목을 기반으로 추천 제목 리스트만 반환합니다.
        (Jikan API 보강은 API 라우터 단에서 수행)
        """
        if self.df is None or not anime_titles:
            return []

        all_recs_titles = set()
        
        # 1. 찜 목록에 있는 각 애니에 대해 하이브리드 추천을 돌립니다.
        for title in anime_titles:
            # 기존 하이브리드 추천 함수 재활용
            recs = self.get_hybrid_recommendations(title, top_n=20) 
            if recs:
                all_recs_titles.update(recs) # 중복 없이 추천 결과를 통합
        
        # 2. 찜 목록 원본은 제외하고 리스트로 변환
        final_candidates = [
            t for t in list(all_recs_titles) 
            if t not in anime_titles
        ]
        
        # 3. 상위 N개만 반환
        return final_candidates[:top_n]
    
    # 애니 상세정보 반환
    def get_anime_details_by_id(self, anime_id: int):
        if self.df is None:
            return None
        result = self.df[self.df['anime_id'] == anime_id]

        if not result.empty:
            return result.iloc[0].to_dict()
        
        return None