import 'react-native-gesture-handler';

import { useEffect, useState } from 'react';
import { ActivityIndicator, View } from 'react-native';
import { Stack, useRouter, useSegments } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

import { AuthProvider, useAuth } from '@/contexts/AuthContext';
import { colors } from '@/theme';
import { Routes } from '@/constants/routes';

// Routes that don't require authentication
const PUBLIC_ROUTES = ['', 'login', 'register'];

function RootLayoutContent() {
  const { isAuthenticated, loading } = useAuth();
  const segments = useSegments();
  const router = useRouter();
  const [isNavigationReady, setIsNavigationReady] = useState(false);

  useEffect(() => {
    if (!loading) {
      setIsNavigationReady(true);
    }
  }, [loading]);

  useEffect(() => {
    if (!isNavigationReady) return;

    const currentSegment = segments[0] ?? '';

    // ✅ FIXED: Check against actual route names, not an 'auth' folder
    const isPublicRoute = PUBLIC_ROUTES.includes(currentSegment);

    if (!isAuthenticated && !isPublicRoute) {
      // Not logged in and trying to access a protected route → send to login
      router.replace(Routes.login);
    } else if (isAuthenticated && isPublicRoute) {
      // Already logged in and on a public screen → send to dashboard
      router.replace(Routes.dashboard);
    }
  }, [isAuthenticated, segments, isNavigationReady]);

  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: colors.background }}>
        <ActivityIndicator size="large" color={colors.primary} />
      </View>
    );
  }

  return (
    <>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: colors.background },
          animation: 'slide_from_right',
        }}
      />
    </>
  );
}

export default function RootLayout() {
  return (
    <AuthProvider>
      <RootLayoutContent />
    </AuthProvider>
  );
}