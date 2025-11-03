import React from 'react';
import './FilterBar.css';

function FilterBar({ onFilterChange, currentFilter }) {
  const genres = [
    '전체', '액션', '모험', '코미디', '드라마', '판타지', 
    '로맨스', '미스터리', 'SF', '스릴러', '호러', '스포츠'
  ];

  const sortOptions = [
    { value: 'popular', label: '인기순' },
    { value: 'score', label: '평점순' },
    { value: 'title', label: '제목순' },
  ];

  return (
    <div className="filter-bar">
      <div className="filter-section">
        <label className="filter-label">장르</label>
        <div className="filter-buttons">
          {genres.map((genre) => (
            <button
              key={genre}
              className={`filter-btn ${currentFilter.genre === genre ? 'active' : ''}`}
              onClick={() => onFilterChange({ ...currentFilter, genre })}
            >
              {genre}
            </button>
          ))}
        </div>
      </div>

      <div className="filter-section">
        <label className="filter-label">정렬</label>
        <select
          className="filter-select"
          value={currentFilter.sort}
          onChange={(e) => onFilterChange({ ...currentFilter, sort: e.target.value })}
        >
          {sortOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div className="filter-section">
        <label className="filter-label">최소 평점</label>
        <input
          type="range"
          className="filter-range"
          min="0"
          max="10"
          step="0.5"
          value={currentFilter.minScore}
          onChange={(e) => onFilterChange({ ...currentFilter, minScore: parseFloat(e.target.value) })}
        />
        <span className="filter-value">{currentFilter.minScore}</span>
      </div>
    </div>
  );
}

export default FilterBar;