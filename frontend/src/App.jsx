import React, { useEffect } from 'react';
import { useAuth } from './context/AuthContext.jsx';
import { Navigate } from 'react-router-dom';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext.jsx';
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import useAuthStore from './stores/authStore';

function PrivateRoute({ children }) {
  const user = useAuthStore(state => state.user);
  return user ? children : <Navigate to="/login" />;
}

function App() {
  return (
    <Router>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/" element={<PrivateRoute><Home /></PrivateRoute>} />
        </Routes>
      </AuthProvider>
    </Router>
  );
}

export default App; 