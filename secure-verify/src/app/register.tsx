import { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { ScreenContainer } from '@/components/ui/ScreenContainer';
import { ScreenHeader } from '@/components/ui/ScreenHeader';
import { Routes } from '@/constants/routes';
import { colors, spacing, typography } from '@/theme';

export default function RegisterScreen() {
  const router = useRouter();
  const { register } = useAuth();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const validateInputs = () => {
    if (!name.trim()) {
      setError('Full name is required');
      return false;
    }

    if (name.trim().length < 2) {
      setError('Name must be at least 2 characters');
      return false;
    }

    if (!email.trim()) {
      setError('Email is required');
      return false;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError('Please enter a valid email address');
      return false;
    }

    if (!password) {
      setError('Password is required');
      return false;
    }

    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return false;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return false;
    }

    return true;
  };

  const handleCreate = async () => {
    setError('');

    if (!validateInputs()) {
      return;
    }

    try {
      setLoading(true);
      const result = await register(name, email.toLowerCase(), password);

      if (result.success) {
        // Clear form
        setName('');
        setEmail('');
        setPassword('');
        setConfirmPassword('');
        
        // Navigate to login screen
        router.replace(Routes.login);
      } else {
        setError(result.message || 'Registration failed');
      }
    } catch (err) {
      setError('An unexpected error occurred');
      console.error('Register error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScreenContainer>
      <ScreenHeader showBack onBack={() => router.back()} />

      <View style={styles.header}>
        <Text style={styles.title}>Create your account</Text>
        <Text style={styles.subtitle}>Join thousands who trust Secure Verify</Text>
      </View>

      {error ? <View style={styles.errorBox}>
        <Ionicons name="alert-circle" size={18} color={colors.danger} />
        <Text style={styles.errorText}>{error}</Text>
      </View> : null}

      <View style={styles.form}>
        <Input
          label="Full Name"
          placeholder="Alex Morgan"
          value={name}
          onChangeText={(text) => {
            setName(text);
            setError('');
          }}
          editable={!loading}
        />
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
          placeholder="Min. 8 characters"
          secureTextEntry
          value={password}
          onChangeText={(text) => {
            setPassword(text);
            setError('');
          }}
          editable={!loading}
        />
        <Input
          label="Confirm Password"
          placeholder="Re-enter password"
          secureTextEntry
          value={confirmPassword}
          onChangeText={(text) => {
            setConfirmPassword(text);
            setError('');
          }}
          editable={!loading}
        />
        <Button label="Create Account" onPress={handleCreate} loading={loading} />
      </View>

      <Pressable onPress={() => router.push(Routes.login)} style={styles.switch} disabled={loading}>
        <Text style={styles.switchText}>
          Already have an account? <Text style={styles.link}>Sign in</Text>
        </Text>
      </Pressable>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  header: {
    gap: spacing.sm,
    marginBottom: spacing.lg,
  },
  title: {
    ...typography.title,
  },
  subtitle: {
    ...typography.subtitle,
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
  },
  switch: {
    marginTop: spacing.xl,
    alignItems: 'center',
  },
  switchText: {
    ...typography.bodySmall,
  },
  link: {
    fontWeight: '700',
    color: colors.secondary,
  },
});
