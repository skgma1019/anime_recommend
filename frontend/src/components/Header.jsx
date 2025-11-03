import React from 'react';

function Header({ 
  isLoggedIn, 
  userEmail, 
  currentPage, 
  onNavigate, 
  onLogout,
  onShowLogin 
}) {
  return (
    <nav className="navbar">
      <div className="nav-container">
        <div className="nav-brand" onClick={() => onNavigate('home')}>
          <span className="brand-text">ANIMERECOM</span>
        </div>
        
        <div className="nav-links">
          <button 
            className={currentPage === 'home' ? 'nav-link active' : 'nav-link'}
            onClick={() => onNavigate('home')}
          >
            홈
          </button>
          <button 
            className={currentPage === 'popular' ? 'nav-link active' : 'nav-link'}
            onClick={() => onNavigate('popular')}
          >
            인기
          </button>
          {isLoggedIn && (
            <>
              <button 
                className={currentPage === 'favorites' ? 'nav-link active' : 'nav-link'}
                onClick={() => onNavigate('favorites')}
              >
                즐겨찾기
              </button>
              <button 
                className={currentPage === 'mypage' ? 'nav-link active' : 'nav-link'}
                onClick={() => onNavigate('mypage')}
              >
                MY
              </button>
            </>
          )}
        </div>

        <div className="nav-actions">
          {isLoggedIn ? (
            <div className="user-menu">
              <button className="user-btn">
                <div className="user-avatar">
                  {userEmail && userEmail.charAt(0).toUpperCase()}
                </div>
              </button>
              <div className="user-dropdown">
                <div className="user-info">
                  <p className="user-email">{userEmail}</p>
                </div>
                <button className="dropdown-item" onClick={() => onNavigate('mypage')}>
                  마이페이지
                </button>
                <button className="dropdown-item logout" onClick={onLogout}>
                  로그아웃
                </button>
              </div>
            </div>
          ) : (
            <button className="login-btn" onClick={onShowLogin}>
              로그인
            </button>
          )}
        </div>
      </div>
    </nav>
  );
}

export default Header;