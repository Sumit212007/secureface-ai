/**
 * AuthContext - Manages authentication state and user session
 * Fixed: Replaced AsyncStorage with expo-secure-store (works in Expo Go)
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import * as SecureStore from 'expo-secure-store';
import * as api from '@/services/api';

const STORAGE_KEY = 'secure_verify_user';

export interface User {
  id: number;
  full_name: string;
  email: string;
  created_at: string;
}

export interface AuthContextType {
  user: User | null;
  loading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<{ success: boolean; message: string }>;
  register: (
    fullName: string,
    email: string,
    password: string
  ) => Promise<{ success: boolean; message: string }>;
  logout: () => Promise<void>;
  hydrate: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Load user from SecureStore on app start
  useEffect(() => {
    hydrate();
  }, []);

  const hydrate = async () => {
    try {
      // ✅ FIXED: was AsyncStorage.getItem
      const stored = await SecureStore.getItemAsync(STORAGE_KEY);
      if (stored) {
        const userData = JSON.parse(stored);
        setUser(userData);
      }
    } catch (error) {
      console.error('Hydration error:', error);
    } finally {
      setLoading(false);
    }
  };

  const login = async (email: string, password: string) => {
    try {
      setLoading(true);
      const response = await api.login(email, password);

      if (response.success && response.user) {
        setUser(response.user);
        // ✅ FIXED: was AsyncStorage.setItem
        await SecureStore.setItemAsync(STORAGE_KEY, JSON.stringify(response.user));
        return { success: true, message: 'Login successful' };
      }

      return { success: false, message: response.message };
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, message: 'Login failed' };
    } finally {
      setLoading(false);
    }
  };

  const register = async (fullName: string, email: string, password: string) => {
    try {
      setLoading(true);
      const response = await api.register(fullName, email, password);

      if (response.success && response.user) {
        // Don't auto-login after registration
        return { success: true, message: 'Registration successful' };
      }

      return { success: false, message: response.message };
    } catch (error) {
      console.error('Register error:', error);
      return { success: false, message: 'Registration failed' };
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    try {
      setLoading(true);
      setUser(null);
      // ✅ FIXED: was AsyncStorage.removeItem
      await SecureStore.deleteItemAsync(STORAGE_KEY);
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        isAuthenticated: user !== null,
        login,
        register,
        logout,
        hydrate,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}