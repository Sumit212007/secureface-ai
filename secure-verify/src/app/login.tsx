import { useState } from 'react';
import { Pressable, StyleSheet, Text, View, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '@/contexts/AuthContext';
import { SecureIllustration } from '@/components/illustrations/SecureIllustration';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { ScreenContainer } from '@/components/ui/ScreenContainer';
import { ScreenHeader } from '@/components/ui/ScreenHeader';
import { Routes } from '@/constants/routes';
import { colors, spacing, typography } from '@/theme';

export default function LoginScreen() {
  const router = useRouter();
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const validateInputs = () => {
    if (!email.trim()) {
      setError('Email is required');
      return false;
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError('Please enter a valid email');
      return false;
    }
    if (!password) {
      setError('Password is required');
      return false;
    }
    return true;
  };

  const handleSignIn = async () => {
    setError('');

    if (!validateInputs()) {
      return;
    }

    try {
      setLoading(true);
      const result = await login(email.toLowerCase(), password);

      if (result.success) {
        setEmail('');
        setPassword('');
        router.replace(Routes.dashboard);
      } else {
        setError(result.message || 'Login failed');
      }
    } catch (err) {
      setError('An unexpected error occurred');
      console.error('Login error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleForgotPassword = () => {
    // TODO: Implement password reset
    Alert.alert('Password Reset', 'This feature will be available soon');
  };

  return (
    <ScreenContainer>
      <ScreenHeader showBack onBack={() => router.replace(Routes.onboarding)} />

      <View style={styles.header}>
        <Text style={styles.title}>Welcome back</Text>
        <Text style={styles.subtitle}>Sign in to continue your verification</Text>
      </View>

      <View style={styles.illustration}>
        <SecureIllustration />
      </View>

      {error ? <View style={styles.errorBox}>
        <Ionicons name="alert-circle" size={18} color={colors.danger} />
        <Text style={styles.errorText}>{error}</Text>
      </View> : null}

      <View style={styles.form}>
        <Input
          label="Email"
          placeholder="you@email.com"
          keyboardType="email-address"
          autoCapitalize="none"
          value={email}
          onChangeText={(text) => {
            setEmail(text);
            setError('');
          }}
          editable={!loading}
        />
        <Input
          label="Password"
          placeholder="Enter your password"
          secureTextEntry
          value={password}
          onChangeText={(text) => {
            setPassword(text);
            setError('');
          }}
          editable={!loading}
        />
        <Pressable onPress={handleForgotPassword} style={styles.forgot} disabled={loading}>
          <Text style={styles.forgotText}>Forgot password?</Text>
        </Pressable>
        <Button label="Sign In" onPress={handleSignIn} loading={loading} />
        <View style={styles.divider}>
          <View style={styles.line} />
          <Text style={styles.or}>or</Text>
          <View style={styles.line} />
        </View>
        <Button
          label="Continue with Google"
          variant="outline"
          onPress={() => Alert.alert('Coming Soon', 'Google login will be available soon')}
          icon={<Ionicons name="logo-google" size={20} color={colors.text} />}
          disabled={loading}
        />
      </View>

      <Pressable onPress={() => router.push(Routes.register)} style={styles.switch} disabled={loading}>
        <Text style={styles.switchText}>
          Don&apos;t have an account? <Text style={styles.link}>Create one</Text>
        </Text>
      </Pressable>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  header: {
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  title: {
    ...typography.title,
  },
  subtitle: {
    ...typography.subtitle,
  },
  illustration: {
    alignItems: 'center',
    marginVertical: spacing.md,
  },
  errorBox: {
    flexDirection: 'row',
    backgroundColor: '#fee2e2',
    borderRadius: 8,
    padding: spacing.sm,
    marginBottom: spacing.md,
    gap: spacing.sm,
    alignItems: 'center',
  },
  errorText: {
    color: colors.danger,
    flex: 1,
    fontSize: 14,
  },
  form: {
    gap: spacing.md,
    marginTop: spacing.md,
  },
  // ✅ FIXED: Removed duplicate — keeping the correct final version
  forgot: {
    alignSelf: 'flex-end',
  },
  forgotText: {
    ...typography.bodySmall,
    color: colors.secondary,
    fontWeight: '600',
  },
  // ✅ FIXED: Removed duplicate — keeping the correct final version
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    marginVertical: spacing.sm,
  },
  line: {
    flex: 1,
    height: 1,
    backgroundColor: colors.border,
  },
  or: {
    ...typography.caption,
  },
  // ✅ FIXED: Removed duplicate — keeping the correct final version
  switch: {
    marginTop: spacing.xl,
    alignItems: 'center',
  },
  switchText: {
    ...typography.bodySmall,
  },
  link: {
    color: colors.secondary,
    fontWeight: '700',
  },
});