from recommender import RecommenderService

# 1. 앱이 시작될 때 '단 한 번' RecommenderService를 초기화합니다.
#    (터미널에 "🚀 Recommender Service 초기화..."가 한 번만 떠야 함)
recommender_service_instance = RecommenderService()

# 2. FastAPI의 Depends()가 이 함수를 호출하여
#    미리 생성된 인스턴스를 '재사용'합니다.
def get_recommender_service():
    return recommender_service_instance