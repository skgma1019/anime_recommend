import React, { useState } from 'react';
import * as animeApi from '../api/anime';
import './MyPage.css';

function MyPage({ userEmail, favoriteCount, onLogout }) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [showPasswordChange, setShowPasswordChange] = useState(false);
  const [newPassword, setNewPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleDeleteAccount = async () => {
    setLoading(true);
    try {
      await animeApi.deleteAccount();
      onLogout();
    } catch (error) {
      setMessage('회원 탈퇴에 실패했습니다');
    }
    setLoading(false);
  };

  const handlePasswordChange = async (e) => {
    e.preventDefault();
    if (!newPassword || newPassword.length < 4) {
      setMessage('비밀번호는 4자 이상이어야 합니다');
      return;
    }

    setLoading(true);
    try {
      await animeApi.updatePassword(newPassword);
      setMessage('비밀번호가 변경되었습니다');
      setShowPasswordChange(false);
      setNewPassword('');
    } catch (error) {
      setMessage('비밀번호 변경에 실패했습니다');
    }
    setLoading(false);
  };

  return (
    <div className="mypage">
      <div className="mypage-container">
        <div className="mypage-header">
          <div className="mypage-avatar">
            {userEmail?.charAt(0).toUpperCase()}
          </div>
          <div className="mypage-info">
            <h2 className="mypage-email">{userEmail}</h2>
            <p className="mypage-stat">{favoriteCount}개의 즐겨찾기</p>
          </div>
        </div>

        <div className="mypage-content">
          <div className="mypage-section">
            <h3 className="section-title">계정 관리</h3>
            
            <div className="mypage-actions">
              <button 
                className="mypage-btn primary"
                onClick={() => setShowPasswordChange(!showPasswordChange)}
              >
                비밀번호 변경
              </button>
              
              <button 
                className="mypage-btn logout"
                onClick={onLogout}
              >
                로그아웃
              </button>
              
              <button 
                className="mypage-btn delete"
                onClick={() => setShowDeleteConfirm(true)}
              >
                회원 탈퇴
              </button>
            </div>

            {message && (
              <div className="mypage-message">
                {message}
              </div>
            )}
          </div>

          {showPasswordChange && (
            <div className="password-change-section">
              <h4>새 비밀번호</h4>
              <form onSubmit={handlePasswordChange}>
                <input
                  type="password"
                  placeholder="새 비밀번호 입력"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="password-input"
                />
                <div className="form-actions">
                  <button 
                    type="submit" 
                    className="submit-btn"
                    disabled={loading}
                  >
                    {loading ? '변경 중...' : '변경하기'}
                  </button>
                  <button 
                    type="button"
                    className="cancel-btn"
                    onClick={() => {
                      setShowPasswordChange(false);
                      setNewPassword('');
                    }}
                  >
                    취소
                  </button>
                </div>
              </form>
            </div>
          )}
        </div>

        {showDeleteConfirm && (
          <div className="confirm-dialog">
            <div className="confirm-content">
              <h3>정말 탈퇴하시겠습니까?</h3>
              <p>모든 데이터가 삭제되며 복구할 수 없습니다</p>
              <div className="confirm-actions">
                <button 
                  className="confirm-delete"
                  onClick={handleDeleteAccount}
                  disabled={loading}
                >
                  {loading ? '처리 중...' : '탈퇴'}
                </button>
                <button 
                  className="confirm-cancel"
                  onClick={() => setShowDeleteConfirm(false)}
                >
                  취소
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default MyPage;