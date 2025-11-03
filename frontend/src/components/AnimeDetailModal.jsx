import React, { useState, useEffect } from 'react';
import * as animeApi from '../api/anime';
import './AnimeDetailModal.css';

function AnimeDetailModal({ animeId, onClose, onAddFavorite, isFavorite }) {
  const [anime, setAnime] = useState(null);
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    loadAnimeDetail();
  }, [animeId]);

  const loadAnimeDetail = async () => {
    try {
      const response = await animeApi.getAnimeDetail(animeId);
      setAnime(response.data);
      
      // 추천도 함께 로드
      const recResponse = await animeApi.getRecommendations(response.data.title);
      setRecommendations(recResponse.data.slice(0, 6));
    } catch (error) {
      console.error('상세 정보 로드 실패');
    }
    setLoading(false);
  };

  if (loading) {
    return (
      <div className="detail-modal-overlay" onClick={onClose}>
        <div className="detail-modal">
          <div className="detail-loading">
            <div className="spinner"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!anime) return null;

  return (
    <div className="detail-modal-overlay" onClick={onClose}>
      <div className="detail-modal" onClick={(e) => e.stopPropagation()}>
        <button className="detail-close" onClick={onClose}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>

        <div className="detail-content">
          <div className="detail-hero">
            <div className="detail-image">
              <img src={anime.image_url} alt={anime.title} />
            </div>
            <div className="detail-info">
              <h1 className="detail-title">{anime.title}</h1>
              <div className="detail-meta">
                <span className="detail-score">{anime.score}</span>
                <span className="detail-favorites">{anime.favorites_count?.toLocaleString()} 즐겨찾기</span>
              </div>
              <div className="detail-genres">
                {anime.genres?.split(', ').map((genre, idx) => (
                  <span key={idx} className="genre-badge">{genre}</span>
                ))}
              </div>
              <button 
                className={`detail-favorite-btn ${isFavorite ? 'active' : ''}`}
                onClick={() => onAddFavorite(anime)}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill={isFavorite ? "currentColor" : "none"}>
                  <path d="M12 21.35L10.55 20.03C5.4 15.36 2 12.27 2 8.5C2 5.41 4.42 3 7.5 3C9.24 3 10.91 3.81 12 5.08C13.09 3.81 14.76 3 16.5 3C19.58 3 22 5.41 22 8.5C22 12.27 18.6 15.36 13.45 20.03L12 21.35Z" stroke="currentColor" strokeWidth="2"/>
                </svg>
                {isFavorite ? '즐겨찾기 해제' : '즐겨찾기 추가'}
              </button>
            </div>
          </div>

          {recommendations.length > 0 && (
            <div className="detail-recommendations">
              <h3>비슷한 애니메이션</h3>
              <div className="detail-rec-grid">
                {recommendations.map((rec) => (
                  <div key={rec.anime_id} className="detail-rec-card">
                    <img src={rec.image_url} alt={rec.title} />
                    <p>{rec.title}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AnimeDetailModal;