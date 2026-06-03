import { StyleSheet, Text, View } from 'react-native';

import { colors, radius, spacing, typography } from '@/theme';

interface StatCardProps {
  value: string | number;
  label: string;
  accent?: boolean;
}

export function StatCard({ value, label, accent }: StatCardProps) {
  return (
    <View style={[styles.card, accent && styles.accent]}>
      <Text style={[styles.value, accent && styles.valueAccent]}>{value}</Text>
      <Text style={styles.label}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flex: 1,
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    gap: 4,
  },
  accent: {
    backgroundColor: colors.primaryMuted,
    borderColor: colors.primary,
  },
  value: {
    ...typography.heading,
    fontSize: 22,
  },
  valueAccent: {
    color: colors.secondary,
  },
  label: {
    ...typography.caption,
    textAlign: 'center',
  },
});
