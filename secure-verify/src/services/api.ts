/**
 * API Service - Handles all communication with SecureEdge AI backend
 */

const API_BASE_URL = 'http://192.168.1.108:5000';

export interface User {
  id: number;
  full_name: string;
  email: string;
  created_at: string;
}

export interface AuthResponse {
  success: boolean;
  message: string;
  user?: User;
}

export interface VerificationRecord {
  id: number;
  user_id: number;
  verification_id: string;
  decision: string;
  similarity: number;
  liveness_score: number;
  timestamp: string;
}

export interface HistoryResponse {
  success: boolean;
  history: VerificationRecord[];
}

export interface VerifyResponse {
  success: boolean;
  decision: string;
  identity: string;
  similarity: number;
  liveness_score: number;
  liveness_decision: string;
  face_detected: boolean;
  processing_time_ms: number;
  message: string;
  verification_id?: string;
}

/**
 * Register a new user account
 */
export async function register(
  fullName: string,
  email: string,
  password: string
): Promise<AuthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        full_name: fullName,
        email,
        password,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      return {
        success: false,
        message: error.message || 'Registration failed',
      };
    }

    return await response.json();
  } catch (error) {
    console.error('Register error:', error);
    return {
      success: false,
      message: 'Network error during registration',
    };
  }
}

/**
 * Login user
 */
export async function login(email: string, password: string): Promise<AuthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email,
        password,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      return {
        success: false,
        message: error.message || 'Login failed',
      };
    }

    return await response.json();
  } catch (error) {
    console.error('Login error:', error);
    return {
      success: false,
      message: 'Network error during login',
    };
  }
}

/**
 * Get verification history for a user
 */
export async function getHistory(userId: number): Promise<HistoryResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/history/${userId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      return {
        success: false,
        history: [],
      };
    }

    return await response.json();
  } catch (error) {
    console.error('Get history error:', error);
    return {
      success: false,
      history: [],
    };
  }
}

/**
 * Verify face image and optionally save to history
 */
export async function verifyFace(
  imageUri: string,
  fileName: string,
  userId?: number
): Promise<VerifyResponse> {
  try {
    const formData = new FormData();
    formData.append('image', {
      uri: imageUri,
      name: fileName || 'selfie.jpg',
      type: 'image/jpeg',
    } as any);

    if (userId) {
      formData.append('user_id', userId.toString());
    }

    const response = await fetch(`${API_BASE_URL}/verify`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      return {
        success: false,
        decision: 'DENY',
        identity: '',
        similarity: 0,
        liveness_score: 0,
        liveness_decision: '',
        face_detected: false,
        processing_time_ms: 0,
        message: error.message || 'Verification failed',
      };
    }

    return await response.json();
  } catch (error) {
    console.error('Verify face error:', error);
    return {
      success: false,
      decision: 'DENY',
      identity: '',
      similarity: 0,
      liveness_score: 0,
      liveness_decision: '',
      face_detected: false,
      processing_time_ms: 0,
      message: 'Network error during verification',
    };
  }
}
