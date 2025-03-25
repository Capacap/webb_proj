import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const authStore = create(
  persist(
    (set) => ({
      user: null,
      token: null,
      error: null,
      isLoading: false,
      
      login: async (email, password) => {
        set({ isLoading: true, error: null })
        try {
          const formData = new URLSearchParams();
          formData.append('username', email);  // FastAPI OAuth2 expects 'username'
          formData.append('password', password);
          
          const response = await fetch('/api/auth/token', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData,
          })
          const data = await response.json()
          
          if (!response.ok) throw new Error(data.detail || 'Login failed')
          
          // Store the token in localStorage
          localStorage.setItem('token', data.access_token);
          
          set({ 
            user: { email },
            token: data.access_token,  // Store token in state
            isLoading: false 
          })
        } catch (error) {
          set({ error: error.message, isLoading: false })
        }
      },

      logout: () => {
        localStorage.removeItem('token');
        set({ user: null, error: null })
      },

      register: async (userData) => {
        set({ isLoading: true, error: null })
        try {
          const response = await fetch('/api/auth/user/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(userData),
          })
          const data = await response.json()
          
          if (!response.ok) throw new Error(data.detail || 'Registration failed')
          
          set({ isLoading: false })
          return data  // Return the response so we can redirect in the Register component
        } catch (error) {
          set({ error: error.message, isLoading: false })
          throw error  // Re-throw to handle in the Register component
        }
      }
    }),
    {
      name: 'auth-storage', // unique name
      getStorage: () => localStorage, // (optional) by default, 'localStorage' is used
    }
  )
)

console.log("Persisted auth state:", JSON.parse(localStorage.getItem('auth-storage')));

export default authStore; 