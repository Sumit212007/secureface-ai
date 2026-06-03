import { useEffect, useRef } from 'react';
import { ActivityIndicator, Animated, Image, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { AnimatedCheckRow } from '@/components/verification/AnimatedCheckRow';
import { Button } from '@/components/ui/Button';
import { FACE_VERIFICATION_CHECKS } from '@/constants/verification';
import { PickedImage } from '@/utils/verificationMedia';
import { colors, radius, spacing, typography } from '@/theme';

interface FaceVerificationStepProps {
  selfie: PickedImage | null;
  isCapturing: boolean;
  isVerifying: boolean;
  completedCheckCount: number;
  verificationComplete: boolean;
  error: string | null;
  onCapture: () => void;
  onRetry: () => void;
}

const LOG = '[FaceVerificationStep]';

export function FaceVerificationStep({
  selfie,
  isCapturing,
  isVerifying,
  completedCheckCount,
  verificationComplete,
  error,
  onCapture,
  onRetry,
}: FaceVerificationStepProps) {
  const successScale = useRef(new Animated.Value(0.9)).current;
  const successOpacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    console.log(LOG, 'state', {
      hasSelfie: selfie !== null,
      selfieUri: selfie?.uri?.slice(0, 80) ?? null,
      isCapturing,
      isVerifying,
      completedCheckCount,
      verificationComplete,
      error,
    });
  }, [selfie, isCapturing, isVerifying, completedCheckCount, verificationComplete, error]);

  useEffect(() => {
    if (verificationComplete) {
      Animated.parallel([
        Animated.spring(successScale, { toValue: 1, friction: 5, useNativeDriver: true }),
        Animated.timing(successOpacity, { toValue: 1, duration: 350, useNativeDriver: true }),
      ]).start();
    } else {
      successScale.setValue(0.9);
      successOpacity.setValue(0);
    }
  }, [verificationComplete, successOpacity, successScale]);

  const showChecks = selfie !== null && (isVerifying || verificationComplete);

  return (
    <View style={styles.stepContent}>
      <Text style={styles.stepTitle}>Face Verification & Liveness</Text>
      <Text style={styles.stepDesc}>
        Take a clear selfie. We&apos;ll run a quick liveness check — no real AI, demo mode only.
      </Text>

      {selfie ? (
        <View style={styles.previewWrap}>
          <Image source={{ uri: selfie.uri }} style={styles.selfiePreview} resizeMode="cover" />
          {isVerifying && !verificationComplete ? (
            <View style={styles.verifyingOverlay}>
              <ActivityIndicator size="large" color={colors.primary} />
              <Text style={styles.verifyingText}>Analyzing…</Text>
            </View>
          ) : null}
        </View>
      ) : (
        <View style={styles.placeholder}>
          <Ionicons name="person-circle-outline" size={64} color={colors.textLight} />
          <Text style={styles.placeholderText}>No selfie captured yet</Text>
        </View>
      )}

      {!selfie && (
        <Button
          label={isCapturing ? 'Opening camera…' : 'Open Camera'}
          variant="secondary"
          onPress={onCapture}
          loading={isCapturing}
          icon={<Ionicons name="camera-outline" size={20} color={colors.surface} />}
        />
      )}

      {error ? (
        <View style={styles.errorBox}>
          <Ionicons name="alert-circle" size={20} color={colors.error} />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : null}

      {selfie && !verificationComplete && (
        <Button
          label="Retry selfie"
          variant="outline"
          onPress={onRetry}
          disabled={isVerifying || isCapturing}
        />
      )}

      {showChecks && (
        <View style={styles.checkList}>
          <Text style={styles.checkTitle}>
            {verificationComplete ? 'All checks passed' : 'Running verification…'}
          </Text>
          {FACE_VERIFICATION_CHECKS.map((label, index) => (
            <AnimatedCheckRow
              key={label}
              label={label}
              completed={index < completedCheckCount}
              pending={index === completedCheckCount && isVerifying}
            />
          ))}
        </View>
      )}

      {verificationComplete && (
        <Animated.View
          style={[
            styles.completeBanner,
            { opacity: successOpacity, transform: [{ scale: successScale }] },
          ]}>
          <Ionicons name="checkmark-circle" size={28} color={colors.success} />
          <Text style={styles.completeText}>Face verification & liveness confirmed</Text>
        </Animated.View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  stepContent: {
    alignItems: 'center',
    gap: spacing.md,
  },
  stepTitle: {
    ...typography.heading,
    textAlign: 'center',
  },
  stepDesc: {
    ...typography.bodySmall,
    textAlign: 'center',
    paddingHorizontal: spacing.sm,
  },
  placeholder: {
    width: '100%',
    height: 180,
    borderRadius: radius.xl,
    backgroundColor: colors.primaryMuted,
    borderWidth: 2,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
  },
  placeholderText: {
    ...typography.bodySmall,
  },
  previewWrap: {
    alignSelf: 'stretch',
    height: 200,
    borderRadius: radius.xl,
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: colors.primary,
  },
  selfiePreview: {
    width: '100%',
    height: '100%',
  },
  verifyingOverlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(17, 17, 17, 0.45)',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
  },
  verifyingText: {
    ...typography.label,
    color: colors.surface,
  },
  errorBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    alignSelf: 'stretch',
    padding: spacing.md,
    backgroundColor: '#FFEBE9',
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.error,
  },
  errorText: {
    ...typography.bodySmall,
    color: colors.error,
    flex: 1,
  },
  checkList: {
    alignSelf: 'stretch',
    gap: spacing.sm,
  },
  checkTitle: {
    ...typography.label,
    marginBottom: spacing.xs,
  },
  completeBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    alignSelf: 'stretch',
    padding: spacing.md,
    backgroundColor: '#E8F9EE',
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.success,
  },
  completeText: {
    ...typography.bodySmall,
    flex: 1,
    fontWeight: '600',
    color: colors.success,
  },
});
