import { StyleSheet, Text, View } from 'react-native';
import { useRouter } from 'expo-router';

import { AppLogo } from '@/components/AppLogo';
import { IdentityIllustration } from '@/components/illustrations/IdentityIllustration';
import { Button } from '@/components/ui/Button';
import { ScreenContainer } from '@/components/ui/ScreenContainer';
import { Routes } from '@/constants/routes';
import { colors, spacing, typography } from '@/theme';

export default function OnboardingScreen() {
  const router = useRouter();

  return (
    <ScreenContainer scroll={false} contentStyle={styles.container}>
      <View style={styles.top}>
        <AppLogo size="lg" />
      </View>

      <View style={styles.hero}>
        <IdentityIllustration />
        <Text style={styles.title}>Secure Identity Verification</Text>
        <Text style={styles.subtitle}>
          Verify your identity in minutes with bank-grade security. Fast, friendly, and fully
          encrypted.
        </Text>
      </View>

      <View style={styles.actions}>
        <Button label="Create Account" onPress={() => router.push(Routes.register)} />
        <Button
          label="Login"
          variant="outline"
          onPress={() => router.push(Routes.login)}
        />
      </View>

      <Text style={styles.footer}>
        By continuing, you agree to our Terms of Service and Privacy Policy.
      </Text>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    justifyContent: 'space-between',
    paddingTop: spacing.lg,
    paddingBottom: spacing.lg,
  },
  top: {
    alignItems: 'center',
  },
  hero: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.lg,
    paddingVertical: spacing.xl,
  },
  title: {
    ...typography.hero,
    textAlign: 'center',
    paddingHorizontal: spacing.md,
  },
  subtitle: {
    ...typography.subtitle,
    textAlign: 'center',
    paddingHorizontal: spacing.md,
  },
  actions: {
    gap: spacing.md,
  },
  footer: {
    ...typography.caption,
    textAlign: 'center',
    marginTop: spacing.lg,
    paddingHorizontal: spacing.xl,
  },
});
