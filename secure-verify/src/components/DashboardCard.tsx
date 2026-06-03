import { StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { Card } from '@/components/ui/Card';
import { colors, spacing, typography } from '@/theme';

interface DashboardCardProps {
  title: string;
  subtitle: string;
  icon: keyof typeof Ionicons.glyphMap;
  onPress: () => void;
  variant?: 'default' | 'primary' | 'dark';
}

export function DashboardCard({
  title,
  subtitle,
  icon,
  onPress,
  variant = 'default',
}: DashboardCardProps) {
  const isDark = variant === 'dark';

  return (
    <Card onPress={onPress} variant={variant} style={styles.card}>
      <View style={[styles.iconWrap, isDark && styles.iconWrapDark]}>
        <Ionicons name={icon} size={26} color={isDark ? colors.primary : colors.secondary} />
      </View>
      <Text style={[styles.title, isDark && styles.titleDark]}>{title}</Text>
      <Text style={[styles.subtitle, isDark && styles.subtitleDark]}>{subtitle}</Text>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    minHeight: 130,
    justifyContent: 'space-between',
  },
  iconWrap: {
    width: 48,
    height: 48,
    borderRadius: 16,
    backgroundColor: colors.primaryMuted,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.md,
  },
  iconWrapDark: {
    backgroundColor: 'rgba(199, 244, 100, 0.15)',
  },
  title: {
    ...typography.heading,
    fontSize: 17,
  },
  titleDark: {
    color: colors.surface,
  },
  subtitle: {
    ...typography.caption,
    marginTop: 4,
  },
  subtitleDark: {
    color: colors.textLight,
  },
});
