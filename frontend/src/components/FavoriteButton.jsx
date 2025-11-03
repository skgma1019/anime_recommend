import React from 'react';
import './FavoriteButton.css';

function FavoriteButton({ isFavorite, onClick, loading }) {
  return (
    <button 
      className={`favorite-button ${isFavorite ? 'active' : ''} ${loading ? 'loading' : ''}`}
      onClick={onClick}
      disabled={loading}
    >
      <svg 
        className="favorite-icon" 
        width="20" 
        height="20" 
        viewBox="0 0 24 24" 
        fill={isFavorite ? "currentColor" : "none"}
      >
        <path 
          d="M12 21.35L10.55 20.03C5.4 15.36 2 12.27 2 8.5C2 5.41 4.42 3 7.5 3C9.24 3 10.91 3.81 12 5.08C13.09 3.81 14.76 3 16.5 3C19.58 3 22 5.41 22 8.5C22 12.27 18.6 15.36 13.45 20.03L12 21.35Z" 
          stroke="currentColor" 
          strokeWidth="2"
        />
      </svg>
      <span className="favorite-text">
        {loading ? '처리 중...' : (isFavorite ? '즐겨찾기 해제' : '즐겨찾기')}
      </span>
    </button>
  );
}

export default FavoriteButton;