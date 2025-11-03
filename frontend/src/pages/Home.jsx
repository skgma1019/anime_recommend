import React from 'react';
import SearchBar from '../components/SearchBar';
import RecommendSection from '../components/RecommendSection';
import AnimeList from '../components/AnimeList';
import './Home.css';

function Home({ 
  popularAnimes,
  recommendations,
  isLoggedIn,
  isFavorite,
  onSearch,
  onAddFavorite,
  onRemoveFavorite,
  onGetRecommendations,
  onShowDetail,
  onShowLogin,
  loading
}) {
  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">
            AI 기반 애니메이션 추천
          </h1>
          <p className="hero-subtitle">
            당신의 취향을 분석하여 완벽한 애니메이션을 찾아드립니다
          </p>
          {!isLoggedIn && (
            <button className="hero-cta" onClick={onShowLogin}>
              시작하기
            </button>
          )}
        </div>
      </section>

      {/* Search Bar */}
      <SearchBar onSearch={onSearch} />

      {/* Personal Recommendations */}
      {isLoggedIn && recommendations.length > 0 && (
        <RecommendSection
          recommendations={recommendations}
          isFavorite={isFavorite}
          onAddFavorite={onAddFavorite}
          onRemoveFavorite={onRemoveFavorite}
          onGetRecommendations={onGetRecommendations}
          onShowDetail={onShowDetail}
        />
      )}

      {/* Popular Animes */}
      <section className="popular-section">
        <div className="section-header">
          <h2 className="section-title">인기 애니메이션</h2>
          <p className="section-subtitle">지금 가장 인기있는 작품</p>
        </div>
        <AnimeList
          animes={popularAnimes}
          isFavorite={isFavorite}
          onAddFavorite={onAddFavorite}
          onRemoveFavorite={onRemoveFavorite}
          onGetRecommendations={onGetRecommendations}
          onShowDetail={onShowDetail}
          loading={loading}
        />
      </section>
    </div>
  );
}

export default Home;