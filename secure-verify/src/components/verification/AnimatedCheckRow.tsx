import { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { colors, radius, spacing, typography } from '@/theme';

interface AnimatedCheckRowProps {
  label: string;
  completed: boolean;
  pending: boolean;
}

export function AnimatedCheckRow({ label, completed, pending }: AnimatedCheckRowProps) {
  const opacity = useRef(new Animated.Value(completed ? 1 : 0.4)).current;
  const scale = useRef(new Animated.Value(completed ? 1 : 0.96)).current;

  useEffect(() => {
    if (completed) {
      Animated.parallel([
        Animated.timing(opacity, { toValue: 1, duration: 280, useNativeDriver: true }),
        Animated.spring(scale, { toValue: 1, friction: 6, useNativeDriver: true }),
      ]).start();
    }
  }, [completed, opacity, scale]);

  return (
    <Animated.View
      style={[
        styles.row,
        completed && styles.rowComplete,
        { opacity, transform: [{ scale }] },
      ]}>
      <View style={[styles.icon, completed && styles.iconComplete]}>
        {completed ? (
          <Ionicons name="checkmark" size={16} color={colors.surface} />
        ) : pending ? (
          <View style={styles.pendingDot} />
        ) : (
          <View style={styles.emptyDot} />
        )}
      </View>
      <Text style={[styles.label, completed && styles.labelComplete]}>{label}</Text>
      {completed && <Ionicons name="checkmark-circle" size={20} color={colors.success} />}
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    padding: spacing.md,
    backgroundColor: colors.background,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  rowComplete: {
    backgroundColor: '#E8F9EE',
    borderColor: colors.success,
  },
  icon: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconComplete: {
    backgroundColor: colors.success,
  },
  emptyDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.textLight,
  },
  pendingDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.primary,
  },
  label: {
    ...typography.body,
    flex: 1,
    color: colors.textMuted,
  },
  labelComplete: {
    color: colors.text,
    fontWeight: '600',
  },
});
