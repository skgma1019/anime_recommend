import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000';

// Axios 인스턴스 생성
const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 요청 인터셉터
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 응답 인터셉터
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // 401 에러 시 자동으로 토큰 삭제
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      delete api.defaults.headers.common['Authorization'];
      
      if (!window.location.pathname.includes('/login')) {
        console.log('토큰이 만료되었습니다. 다시 로그인해주세요.');
      }
    }
    return Promise.reject(error);
  }
);

// 토큰 설정
export const setAuthToken = (token) => {
  if (token) {
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    localStorage.setItem('token', token);
  } else {
    delete api.defaults.headers.common['Authorization'];
    localStorage.removeItem('token');
  }
};

// 페이지 로드 시 토큰 복원
const token = localStorage.getItem('token');
if (token) {
  setAuthToken(token);
}

// 1. 기본 GET
export const getHome = () => api.get('/');

// 2. 검색
export const searchAnime = (keyword) => 
  api.get(`/animes/search?keyword=${keyword}`);

// 3. 애니 목록
export const getAnimeList = (limit = 20) => 
  api.get(`/animes?limit=${limit}`);

// 4. 애니 추천
export const getRecommendations = (title) => 
  api.get(`/animes/recommend?title=${title}`);

// 5. 인기 애니
export const getPopularAnimes = () => 
  api.get('/animes/popular');

// 6. 회원가입
export const register = (email, password) => 
  api.post('/users/register', { email, password });

// 7. 로그인
export const login = async (email, password) => {
  const formData = new FormData();
  formData.append('username', email);
  formData.append('password', password);
  
  const response = await axios.post(`${API_BASE}/users/login`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  
  if (response.data.access_token) {
    setAuthToken(response.data.access_token);
  }
  
  return response;
};

// 8. 로그아웃
export const logout = () => {
  setAuthToken(null);
};

// 9. 정보 수정
export const updatePassword = (newPassword) => 
  api.put('/users/me', { new_password: newPassword });

// 10. 회원 탈퇴
export const deleteAccount = () => 
  api.delete('/users/me');

// 11. 본인 확인
export const getMyInfo = () => 
  api.get('/users/me');

// 12. 즐겨찾기 추가
export const addFavorite = (animeId, title, imageUrl) => {
  console.log('🔵 API 호출 - 즐겨찾기 추가:', { animeId, title, imageUrl });
  return api.post('/users/me/favorites', {
    anime_id: animeId,
    title: title,
    image_url: imageUrl
  });
};

// 13. 즐겨찾기 목록
export const getFavorites = () => 
  api.get('/users/me/favorites');

// 14. 개인 맞춤 추천
export const getPersonalRecommendations = () => 
  api.get('/users/me/recommendations');

// 15. 애니 상세 정보
export const getAnimeDetail = (animeId) => 
  api.get(`/animes/${animeId}`);

// 16. 즐겨찾기 삭제
export const removeFavorite = (animeId) => {
  console.log('🔴 API 호출 - 즐겨찾기 삭제:', animeId);
  console.log('🔴 삭제 URL:', `/users/me/favorites/${animeId}`);
  return api.delete(`/users/me/favorites/${animeId}`);
};

// 17. 피드백 제출
export const submitFeedback = (type, satisfied, text) => 
  api.post('/users/me/feedback', {
    recommendation_type: type,
    is_satisfied: satisfied,
    feedback_text: text
  });

export default api;