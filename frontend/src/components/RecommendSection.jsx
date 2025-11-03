import React from 'react';
import AnimeList from './AnimeList';
import './RecommendSection.css';

function RecommendSection({ 
  recommendations,
  isFavorite,
  onAddFavorite,
  onRemoveFavorite,
  onGetRecommendations,
  onShowDetail,
  loading 
}) {
  if (loading) {
    return (
      <div className="recommend-section">
        <div className="recommend-loading">
          <div className="spinner"></div>
          <p>추천 생성 중...</p>
        </div>
      </div>
    );
  }

  if (!recommendations || recommendations.length === 0) {
    return null;
  }

  return (
    <section className="recommend-section">
      <div className="section-header">
        <h2 className="section-title">당신을 위한 추천</h2>
        <p className="section-subtitle">AI가 선택한 맞춤 애니메이션</p>
      </div>
      <AnimeList
        animes={recommendations}
        isFavorite={isFavorite}
        onAddFavorite={onAddFavorite}
        onRemoveFavorite={onRemoveFavorite}
        onGetRecommendations={onGetRecommendations}
        onShowDetail={onShowDetail}
      />
    </section>
  );
}

export default RecommendSection;