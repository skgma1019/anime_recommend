import React, { useState } from 'react';
import './AnimeCard.css';

function AnimeCard({ 
  anime, 
  isFavorite, 
  onAddFavorite, 
  onRemoveFavorite, 
  onGetRecommendations,
  onShowDetail 
}) {
  const [showActions, setShowActions] = useState(false);

  return (
    <div 
      className="anime-card"
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
      onClick={() => onShowDetail && onShowDetail(anime.anime_id)}
    >
      <div className="card-image">
        <img 
          src={anime.image_url || anime.images?.jpg?.image_url} 
          alt={anime.title}
          loading="lazy"
        />
        <div className={`card-overlay ${showActions ? 'show' : ''}`}>
          <div className="card-actions">
            <button 
              className="action-btn"
              onClick={(e) => {
                e.stopPropagation();
                isFavorite ? onRemoveFavorite(anime.anime_id) : onAddFavorite(anime);
              }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill={isFavorite ? "currentColor" : "none"}>
                <path d="M12 21.35L10.55 20.03C5.4 15.36 2 12.27 2 8.5C2 5.41 4.42 3 7.5 3C9.24 3 10.91 3.81 12 5.08C13.09 3.81 14.76 3 16.5 3C19.58 3 22 5.41 22 8.5C22 12.27 18.6 15.36 13.45 20.03L12 21.35Z" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </button>
            {onGetRecommendations && (
              <button 
                className="action-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  onGetRecommendations(anime.title);
                }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M13 10V3L4 14H11V21L20 10H13Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
      <div className="card-content">
        <h3 className="card-title">{anime.title}</h3>
        <div className="card-meta">
          <span className="card-score">{anime.score || 'N/A'}</span>
          {anime.favorites_count && (
            <span className="card-favorites">{anime.favorites_count.toLocaleString()}</span>
          )}
        </div>
        {anime.genres && (
          <p className="card-genres">{anime.genres}</p>
        )}
      </div>
    </div>
  );
}

export default AnimeCard;