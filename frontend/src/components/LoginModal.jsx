import React, { useState } from 'react';
import './LoginModal.css';

function LoginModal({ onClose, onLogin, onRegister }) {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isLogin) {
        await onLogin(email, password);
      } else {
        await onRegister(email, password);
        setIsLogin(true);
        setPassword('');
        setError('회원가입이 완료되었습니다. 로그인해주세요.');
      }
    } catch (err) {
      setError(err.message);
    }

    setLoading(false);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>

        <div className="modal-header">
          <h2>{isLogin ? '로그인' : '회원가입'}</h2>
          <p>{isLogin ? 'AI 추천을 받아보세요' : '새로운 계정 만들기'}</p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label>이메일</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="your@email.com"
              required
            />
          </div>

          <div className="form-group">
            <label>비밀번호</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
            />
          </div>

          {error && <div className="form-error">{error}</div>}

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? '처리 중...' : (isLogin ? '로그인' : '회원가입')}
          </button>
        </form>

        <div className="modal-footer">
          <button 
            className="toggle-auth"
            onClick={() => {
              setIsLogin(!isLogin);
              setError('');
            }}
          >
            {isLogin ? '계정이 없으신가요?' : '이미 계정이 있으신가요?'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default LoginModal;