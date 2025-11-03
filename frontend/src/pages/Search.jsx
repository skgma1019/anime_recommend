import React from 'react';
import SearchBar from '../components/SearchBar';
import AnimeList from '../components/AnimeList';
import './Search.css';

function Search({ 
  searchKeyword,
  searchResults,
  isFavorite,
  onSearch,
  onAddFavorite,
  onRemoveFavorite,
  onGetRecommendations,
  onShowDetail,
  loading
}) {
  return (
    <div className="search-page">
      <div className="search-header">
        <h1 className="page-title">검색</h1>
        <SearchBar onSearch={onSearch} />
      </div>

      {searchResults.length > 0 && (
        <div className="search-results">
          <div className="results-info">
            <h2 className="results-title">
              "{searchKeyword}" 검색 결과
            </h2>
            <p className="results-count">{searchResults.length}개</p>
          </div>

          <AnimeList
            animes={searchResults}
            isFavorite={isFavorite}
            onAddFavorite={onAddFavorite}
            onRemoveFavorite={onRemoveFavorite}
            onGetRecommendations={onGetRecommendations}
            onShowDetail={onShowDetail}
            loading={loading}
          />
        </div>
      )}

      {!loading && searchResults.length === 0 && searchKeyword && (
        <div className="no-results">
          <div className="no-results-icon">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none">
              <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
          <h3>검색 결과가 없습니다</h3>
          <p>다른 키워드로 검색해보세요</p>
        </div>
      )}

      {!searchKeyword && (
        <div className="search-empty">
          <div className="search-empty-icon">
            <svg width="80" height="80" viewBox="0 0 24 24" fill="none">
              <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
          <h3>애니메이션을 검색해보세요</h3>
          <p>제목, 장르 등으로 원하는 작품을 찾아보세요</p>
        </div>
      )}
    </div>
  );
}

export default Search;