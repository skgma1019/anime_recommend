import React from 'react';
import AnimeCard from './AnimeCard';
import './AnimeList.css';

function AnimeList({ 
  animes, 
  isFavorite,
  onAddFavorite, 
  onRemoveFavorite,
  onGetRecommendations,
  onShowDetail,
  loading 
}) {
  if (loading) {
    return (
      <div className="anime-list-loading">
        <div className="spinner"></div>
        <p>로딩 중...</p>
      </div>
    );
  }

  if (!animes || animes.length === 0) {
    return (
      <div className="anime-list-empty">
        <p>애니메이션이 없습니다</p>
      </div>
    );
  }

  return (
    <div className="anime-list">
      <div className="anime-grid">
        {animes.map((anime) => (
          <AnimeCard
            key={anime.anime_id || anime.mal_id}
            anime={anime}
            isFavorite={isFavorite(anime.anime_id || anime.mal_id)}
            onAddFavorite={onAddFavorite}
            onRemoveFavorite={onRemoveFavorite}
            onGetRecommendations={onGetRecommendations}
            onShowDetail={onShowDetail}
          />
        ))}
      </div>
    </div>
  );
}

export default AnimeList;