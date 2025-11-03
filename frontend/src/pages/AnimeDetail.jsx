import React, { useState, useEffect } from 'react';
import * as animeApi from '../api/anime';
import AnimeList from '../components/AnimeList';
import FavoriteButton from '../components/FavoriteButton';
import './AnimeDetail.css';

function AnimeDetail({ 
  animeId,
  isFavorite,
  isLoggedIn,
  onAddFavorite,
  onRemoveFavorite,
  onGetRecommendations,
  onShowDetail,
  onShowLogin,
  onClose
}) {
  const [anime, setAnime] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [favoriteLoading, setFavoriteLoading] = useState(false);

  useEffect(() => {
    loadAnimeDetail();
  }, [animeId]);

  const loadAnimeDetail = async () => {
    setLoading(true);
    try {
      const response = await animeApi.getAnimeDetail(animeId);
      setAnime(response.data);
      
      // 추천 로드
      const recResponse = await animeApi.getRecommendations(response.data.title);
      setRecommendations(recResponse.data.slice(0, 6));
    } catch (error) {
      console.error('상세 정보 로드 실패:', error);
    }
    setLoading(false);
  };

  const handleFavoriteClick = async () => {
    if (!isLoggedIn) {
      onShowLogin();
      return;
    }

    setFavoriteLoading(true);
    try {
      if (isFavorite) {
        await onRemoveFavorite(anime.anime_id);
      } else {
        await onAddFavorite(anime);
      }
    } catch (error) {
      console.error('즐겨찾기 처리 실패');
    }
    setFavoriteLoading(false);
  };

  if (loading) {
    return (
      <div className="detail-page">
        <div className="detail-loading">
          <div className="spinner"></div>
          <p>로딩 중...</p>
        </div>
      </div>
    );
  }

  if (!anime) {
    return (
      <div className="detail-page">
        <div className="detail-error">
          <h3>애니메이션을 찾을 수 없습니다</h3>
        </div>
      </div>
    );
  }

  return (
    <div className="detail-page">
      {onClose && (
        <button className="detail-back" onClick={onClose}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          뒤로가기
        </button>
      )}

      <div className="detail-container">
        {/* Hero Section */}
        <div className="detail-hero">
          <div className="detail-image-wrapper">
            <div className="detail-image">
              <img src={anime.image_url} alt={anime.title} />
            </div>
          </div>

          <div className="detail-info">
            <h1 className="detail-title">{anime.title}</h1>

            <div className="detail-meta">
              <div className="meta-item">
                <span className="meta-label">평점</span>
                <span className="meta-value score">{anime.score}</span>
              </div>
              <div className="meta-item">
                <span className="meta-label">즐겨찾기</span>
                <span className="meta-value">{anime.favorites_count?.toLocaleString()}</span>
              </div>
            </div>

            {anime.genres && (
              <div className="detail-genres">
                {anime.genres.split(', ').map((genre, idx) => (
                  <span key={idx} className="genre-badge">{genre}</span>
                ))}
              </div>
            )}

            <div className="detail-actions">
              <FavoriteButton
                isFavorite={isFavorite}
                onClick={handleFavoriteClick}
                loading={favoriteLoading}
              />
              {onGetRecommendations && (
                <button 
                  className="recommend-btn"
                  onClick={() => onGetRecommendations(anime.title)}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M13 10V3L4 14H11V21L20 10H13Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  비슷한 작품 추천
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Recommendations */}
        {recommendations.length > 0 && (
          <div className="detail-recommendations">
            <h2 className="recommendations-title">비슷한 애니메이션</h2>
            <AnimeList
              animes={recommendations}
              isFavorite={(id) => false}
              onAddFavorite={onAddFavorite}
              onRemoveFavorite={onRemoveFavorite}
              onGetRecommendations={onGetRecommendations}
              onShowDetail={onShowDetail}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default AnimeDetail;