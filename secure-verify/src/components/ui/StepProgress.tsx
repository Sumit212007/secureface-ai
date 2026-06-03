import { StyleSheet, Text, View } from 'react-native';

import { VERIFICATION_STEPS } from '@/constants/app';
import { colors, radius, spacing, typography } from '@/theme';

interface StepProgressProps {
  currentStep: number;
}

export function StepProgress({ currentStep }: StepProgressProps) {
  const total = VERIFICATION_STEPS.length;
  const progress = (currentStep / total) * 100;

  return (
    <View style={styles.wrapper}>
      <View style={styles.header}>
        <Text style={styles.stepLabel}>
          Step {currentStep} of {total}
        </Text>
        <Text style={styles.percent}>{Math.round(progress)}%</Text>
      </View>
      <View style={styles.track}>
        <View style={[styles.fill, { width: `${progress}%` }]} />
      </View>
      <View style={styles.dots}>
        {VERIFICATION_STEPS.map((step) => {
          const isComplete = step.id < currentStep;
          const isActive = step.id === currentStep;
          return (
            <View key={step.id} style={styles.dotRow}>
              <View
                style={[
                  styles.dot,
                  isComplete && styles.dotComplete,
                  isActive && styles.dotActive,
                ]}>
                {isComplete ? (
                  <Text style={styles.check}>✓</Text>
                ) : (
                  <Text style={[styles.dotNum, isActive && styles.dotNumActive]}>{step.id}</Text>
                )}
              </View>
              {step.id < total && (
                <View style={[styles.connector, isComplete && styles.connectorComplete]} />
              )}
            </View>
          );
        })}
      </View>
      <Text style={styles.title}>{VERIFICATION_STEPS[currentStep - 1]?.title}</Text>
      <Text style={styles.subtitle}>{VERIFICATION_STEPS[currentStep - 1]?.subtitle}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    gap: spacing.md,
    marginBottom: spacing.lg,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  stepLabel: {
    ...typography.label,
    color: colors.textMuted,
  },
  percent: {
    ...typography.label,
    color: colors.secondary,
  },
  track: {
    height: 6,
    backgroundColor: colors.border,
    borderRadius: radius.full,
    overflow: 'hidden',
  },
  fill: {
    height: '100%',
    backgroundColor: colors.primary,
    borderRadius: radius.full,
  },
  dots: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: spacing.sm,
  },
  dotRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dot: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.surface,
    borderWidth: 2,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  dotComplete: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  dotActive: {
    borderColor: colors.secondary,
    backgroundColor: colors.secondary,
  },
  check: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.secondary,
  },
  dotNum: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.textMuted,
  },
  dotNumActive: {
    color: colors.surface,
  },
  connector: {
    width: 28,
    height: 2,
    backgroundColor: colors.border,
    marginHorizontal: 4,
  },
  connectorComplete: {
    backgroundColor: colors.primary,
  },
  title: {
    ...typography.heading,
    textAlign: 'center',
    marginTop: spacing.sm,
  },
  subtitle: {
    ...typography.bodySmall,
    textAlign: 'center',
  },
});
