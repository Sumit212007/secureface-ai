import { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

import { SuccessIllustration } from '@/components/illustrations/SuccessIllustration';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { ScreenContainer } from '@/components/ui/ScreenContainer';
import { Routes } from '@/constants/routes';
import { colors, radius, spacing, typography } from '@/theme';

function formatTimestamp(iso: string | undefined): string {
  if (!iso) {
    return new Date().toLocaleString();
  }
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return new Date().toLocaleString();
  }
  return date.toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  });
}

export default function VerificationSuccessScreen() {
  const router = useRouter();
  const { id, timestamp } = useLocalSearchParams<{ id?: string; timestamp?: string }>();
  const verificationId = id ?? 'SV-2025-0000';
  const verifiedAt = formatTimestamp(timestamp);

  const scale = useRef(new Animated.Value(0.85)).current;
  const opacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.spring(scale, { toValue: 1, friction: 6, useNativeDriver: true }),
      Animated.timing(opacity, { toValue: 1, duration: 400, useNativeDriver: true }),
    ]).start();
  }, [opacity, scale]);

  return (
    <ScreenContainer scroll={false} contentStyle={styles.container}>
      <Animated.View style={{ opacity, transform: [{ scale }] }}>
        <SuccessIllustration />
      </Animated.View>

      <Text style={styles.title}>Verification Complete</Text>
      <Text style={styles.subtitle}>
        Your identity has been verified and approved. Save your verification ID for your records.
      </Text>

      <View style={styles.approvedBadge}>
        <Ionicons name="checkmark-circle" size={22} color={colors.success} />
        <Text style={styles.approvedText}>Approved</Text>
      </View>

      <Card style={styles.idCard}>
        <Text style={styles.idLabel}>Verification ID</Text>
        <Text style={styles.idValue}>{verificationId}</Text>
        <View style={styles.metaRow}>
          <Ionicons name="time-outline" size={18} color={colors.textMuted} />
          <Text style={styles.metaText}>{verifiedAt}</Text>
        </View>
      </Card>

      <View style={styles.actions}>
        <Button
          label="Continue to Dashboard"
          onPress={() => router.replace(Routes.dashboard)}
          icon={<Ionicons name="home-outline" size={20} color={colors.secondary} />}
        />
      </View>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.lg,
    paddingVertical: spacing.xl,
  },
  title: {
    ...typography.hero,
    fontSize: 28,
    textAlign: 'center',
  },
  subtitle: {
    ...typography.subtitle,
    textAlign: 'center',
    paddingHorizontal: spacing.md,
  },
  approvedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: '#E8F9EE',
    borderRadius: radius.full,
    borderWidth: 1.5,
    borderColor: colors.success,
  },
  approvedText: {
    ...typography.label,
    color: colors.success,
    fontSize: 15,
  },
  idCard: {
    alignSelf: 'stretch',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: colors.primaryMuted,
    borderColor: colors.primary,
  },
  idLabel: {
    ...typography.caption,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  idValue: {
    ...typography.title,
    fontSize: 22,
    letterSpacing: 1,
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.sm,
  },
  metaText: {
    ...typography.bodySmall,
    fontWeight: '500',
  },
  actions: {
    alignSelf: 'stretch',
    marginTop: spacing.md,
  },
});
