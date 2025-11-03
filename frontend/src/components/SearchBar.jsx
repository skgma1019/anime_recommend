import React, { useState } from 'react';
import './SearchBar.css';

function SearchBar({ onSearch }) {
  const [keyword, setKeyword] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (keyword.trim()) {
      onSearch(keyword);
    }
  };

  return (
    <form className="search-bar" onSubmit={handleSubmit}>
      <div className="search-input-wrapper">
        <svg className="search-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        </svg>
        <input
          type="text"
          className="search-input"
          placeholder="애니메이션 검색..."
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
        />
        <button type="submit" className="search-submit-btn">
          검색
        </button>
      </div>
    </form>
  );
}

export default SearchBar;