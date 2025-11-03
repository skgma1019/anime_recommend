import React, { useState, useEffect } from 'react';
import './App.css';
import * as animeApi from './api/anime';

// Components
import Header from './components/Header';
import LoginModal from './components/LoginModal';
import AnimeDetailModal from './components/AnimeDetailModal';

// Pages
import Home from './pages/Home';
import Search from './pages/Search';
import MyPage from './pages/MyPage';

import AnimeCard from './components/AnimeCard';

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [selectedAnimeId, setSelectedAnimeId] = useState(null);
  
  // Data states
  const [searchKeyword, setSearchKeyword] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [popularAnimes, setPopularAnimes] = useState([]);
  const [myFavorites, setMyFavorites] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  
  // Filter state
  const [filter, setFilter] = useState({
    genre: '전체',
    sort: 'popular',
    minScore: 0
  });
  
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      setIsLoggedIn(true);
      loadUserInfo();
      loadMyData();
    }
    loadPopularAnimes();
  }, []);
  useEffect(()=> {
    if (currentPage === 'popular') {
      loadPopularAnimes();
    }
  })
  const loadUserInfo = async () => {
    try {
      const response = await animeApi.getMyInfo();
      setUserEmail(response.data.email);
    } catch (error) {
      console.error('사용자 정보 로드 실패', error);
      if (error.response?.status === 401) {
        handleLogout();
      }
    }
  };

  const loadPopularAnimes = async () => {
    try {
      const response = await animeApi.getPopularAnimes();
      setPopularAnimes(response.data);
    } catch (error) {
      console.error('인기 애니 로드 실패', error);
    }
  };

  const loadMyData = async () => {
    try {
      const favs = await animeApi.getFavorites();
      setMyFavorites(favs.data);
      console.log('📋 즐겨찾기 목록 로드:', favs.data);
      
      // 즐겨찾기가 있을 때만 추천 가져오기
      if (favs.data && favs.data.length > 0) {
        try {
          const recs = await animeApi.getPersonalRecommendations();
          setRecommendations(recs.data.recommendations || []);
        } catch (recError) {
          // 404는 정상 (즐겨찾기가 없으면 추천도 없음)
          if (recError.response?.status !== 404) {
            console.error('추천 로드 실패', recError);
          }
          setRecommendations([]);
        }
      } else {
        setRecommendations([]);
      }
    } catch (error) {
      // 404는 무시 (즐겨찾기 없음)
      if (error.response?.status === 404) {
        setRecommendations([]);
        return;
      }
      
      console.error('데이터 로드 실패', error);
      
      if (error.response?.status === 401) {
        handleLogout();
      }
    }
  };

  const handleSearch = async (keyword) => {
    setSearchKeyword(keyword);
    setLoading(true);
    try {
      const response = await animeApi.searchAnime(keyword);
      setSearchResults(response.data);
      setCurrentPage('search');
    } catch (error) {
      console.error('검색 실패', error);
    }
    setLoading(false);
  };

  const handleLogin = async (email, password) => {
    try {
      await animeApi.login(email, password);
      setIsLoggedIn(true);
      setShowLoginModal(false);
      await loadUserInfo();
      await loadMyData();
    } catch (error) {
      throw new Error('로그인에 실패했습니다');
    }
  };

  const handleRegister = async (email, password) => {
    try {
      await animeApi.register(email, password);
      return true;
    } catch (error) {
      throw new Error('회원가입에 실패했습니다');
    }
  };

  const handleLogout = () => {
    animeApi.logout();
    setIsLoggedIn(false);
    setUserEmail('');
    setMyFavorites([]);
    setRecommendations([]);
    setCurrentPage('home');
  };

  const handleAddFavorite = async (anime) => {
    if (!isLoggedIn) {
      setShowLoginModal(true);
      return;
    }

    const animeId = anime.anime_id || anime.mal_id;
    
    console.log('➕ 즐겨찾기 추가 시도:', {
      anime_id: animeId,
      title: anime.title,
      image_url: anime.image_url || anime.images?.jpg?.image_url,
      원본객체: anime
    });

    try {
      await animeApi.addFavorite(
        animeId,
        anime.title,
        anime.image_url || anime.images?.jpg?.image_url
      );
      await loadMyData();
      console.log('✅ 즐겨찾기 추가 성공');
    } catch (error) {
      console.error('❌ 즐겨찾기 추가 실패', error);
      console.error('에러 상세:', error.response?.data);
      
      if (error.response?.status === 401) {
        handleLogout();
        setShowLoginModal(true);
      } else {
        alert(`추가 실패: ${error.response?.data?.detail || '알 수 없는 오류'}`);
      }
    }
  };

  const handleRemoveFavorite = async (animeId) => {
    console.log('🗑️ 즐겨찾기 삭제 시도 anime_id:', animeId);
    console.log('📋 현재 즐겨찾기 목록:', myFavorites);
    
    // 즐겨찾기 목록에서 해당 anime_id 찾기
    const targetFavorite = myFavorites.find(fav => fav.anime_id === animeId);
    console.log('🎯 삭제 대상:', targetFavorite);
    
    try {
      await animeApi.removeFavorite(animeId);
      await loadMyData();
      console.log('✅ 즐겨찾기 삭제 성공');
    } catch (error) {
      console.error('❌ 즐겨찾기 삭제 실패', error);
      console.error('에러 상세:', error.response?.data);
      alert(`삭제 실패: ${error.response?.data?.detail || '알 수 없는 오류'}`);
    }  finally {
      await loadMyData();
    }
  };

  const handleGetRecommendations = async (title) => {
    setLoading(true);
    try {
      const response = await animeApi.getRecommendations(title);
      setRecommendations(response.data.recommendations || []);
      setCurrentPage('recommendations');
    } catch (error) {
      console.error('추천 받기 실패', error);
    }
    setLoading(false);
  };

  const isFavorite = (animeId) => {
    return myFavorites.some(fav => fav.anime_id === animeId);
  };

  const handleShowDetail = (animeId) => {
    setSelectedAnimeId(animeId);
  };

  const handleCloseDetail = () => {
    setSelectedAnimeId(null);
  };

  return (
    <div className="app">
      {/* Header */}
      <Header
        isLoggedIn={isLoggedIn}
        userEmail={userEmail}
        currentPage={currentPage}
        onNavigate={setCurrentPage}
        onLogout={handleLogout}
        onShowLogin={() => setShowLoginModal(true)}
      />

      {/* Main Content */}
      <main className="main-content">
        {loading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
          </div>
        )}

        {/* Home Page */}
        {currentPage === 'home' && (
          <Home
            popularAnimes={popularAnimes}
            recommendations={recommendations}
            isLoggedIn={isLoggedIn}
            isFavorite={isFavorite}
            onSearch={handleSearch}
            onAddFavorite={handleAddFavorite}
            onRemoveFavorite={handleRemoveFavorite}
            onGetRecommendations={handleGetRecommendations}
            onShowDetail={handleShowDetail}
            onShowLogin={() => setShowLoginModal(true)}
          />
        )}

        {/* Search Page */}
        {currentPage === 'search' && (
          <Search
            searchKeyword={searchKeyword}
            searchResults={searchResults}
            filter={filter}
            isFavorite={isFavorite}
            onSearch={handleSearch}
            onFilterChange={setFilter}
            onAddFavorite={handleAddFavorite}
            onRemoveFavorite={handleRemoveFavorite}
            onGetRecommendations={handleGetRecommendations}
            onShowDetail={handleShowDetail}
            loading={loading}
          />
        )}

        {/* Popular Page */}
        {currentPage === 'popular' && (
          <div className="page">
            <section className="section">
              <div className="section-header">
                <h2 className="section-title">인기 애니메이션</h2>
                <p className="section-subtitle">전체 {popularAnimes.length}개</p>
              </div>
              <div className="anime-grid">
                {popularAnimes.map((anime) => (
                  <div key={anime.anime_id} className="anime-card-container"> 
                    {/* 🌟 [수정] AnimeCard 컴포넌트 추가 */}
                    <AnimeCard 
                        anime={anime}
                        isFavorite={isFavorite(anime.anime_id)}
                        onAddFavorite={handleAddFavorite}
                        onRemoveFavorite={handleRemoveFavorite}
                        onGetRecommendations={handleGetRecommendations}
                        onShowDetail={() => handleShowDetail(anime.anime_id)}
                        // AnimeCard에 전달해야 하는 다른 prop이 있다면 여기에 추가하세요.
                    />
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}

        {/* Favorites Page */}
        {currentPage === 'favorites' && (
          <div className="page">
            <section className="section">
              <div className="section-header">
                <h2 className="section-title">즐겨찾기</h2>
                <p className="section-subtitle">전체 {myFavorites.length}개</p>
              </div>
              {myFavorites.length > 0 ? (
                <div className="anime-grid">
                  {/* 🌟 [수정] myFavorites 목록을 AnimeCard로 렌더링 */}
                    {myFavorites.map((favorite) => (
                        <div key={favorite.anime_id} className="anime-card-container">
                            {/* 참고: myFavorites 데이터 구조는 일반 Anime 객체와 조금 다를 수 있습니다. */}
                            {/* 따라서 favorite 객체를 AnimeCard의 props로 전달할 때 구조를 맞춰주어야 합니다. */}
                            <AnimeCard 
                                // 즐겨찾기 목록의 데이터를 AnimeCard가 이해할 수 있는 형식으로 변환하여 전달합니다.
                                anime={{
                                    anime_id: favorite.anime_id,
                                    title: favorite.title,
                                    image_url: favorite.image_url,
                                    // 즐겨찾기 목록에는 score, synopsis 등의 정보가 없을 수 있습니다.
                                    // 필요한 경우 이 객체를 확장하거나, AnimeCard에서 필수 prop을 제거해야 합니다.
                                }}
                                // 즐겨찾기 목록이므로 isFavorite은 항상 true입니다.
                                isFavorite={true} 
                                // 즐겨찾기 페이지의 주된 액션은 '삭제'입니다.
                                onRemoveFavorite={handleRemoveFavorite}
                                // onAddFavorite와 onGetRecommendations는 필요하지 않을 수 있지만, 안전을 위해 전달
                                onAddFavorite={handleAddFavorite}
                                onGetRecommendations={handleGetRecommendations}
                                onShowDetail={() => handleShowDetail(favorite.anime_id)}
                            />
                        </div>
                    ))}
                </div>
              ) : (
                <div className="empty-state">
                  <p>즐겨찾기가 없습니다. 마음에 드는 애니메이션에 하트를 눌러보세요!</p>
                </div>
              )}
            </section>
          </div>
        )}

        {/* My Page */}
        {currentPage === 'mypage' && (
          <MyPage
            userEmail={userEmail}
            favoriteCount={myFavorites.length}
            onLogout={handleLogout}
          />
        )}
      </main>

      {/* Login Modal */}
      {showLoginModal && (
        <LoginModal
          onClose={() => setShowLoginModal(false)}
          onLogin={handleLogin}
          onRegister={handleRegister}
        />
      )}

      {/* Anime Detail Modal */}
      {selectedAnimeId && (
        <AnimeDetailModal
          animeId={selectedAnimeId}
          onClose={handleCloseDetail}
          onAddFavorite={handleAddFavorite}
          isFavorite={isFavorite(selectedAnimeId)}
        />
      )}
    </div>
  );
}

export default App;