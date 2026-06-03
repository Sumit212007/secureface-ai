import { Pressable, StyleSheet, View, ViewStyle } from 'react-native';

import { colors, radius, shadows, spacing } from '@/theme';

interface CardProps {
  children: React.ReactNode;
  onPress?: () => void;
  style?: ViewStyle;
  variant?: 'default' | 'primary' | 'dark';
}

export function Card({ children, onPress, style, variant = 'default' }: CardProps) {
  const content = (
    <View style={[styles.card, styles[variant], shadows.card, style]}>{children}</View>
  );

  if (onPress) {
    return (
      <Pressable onPress={onPress} style={({ pressed }) => pressed && styles.pressed}>
        {content}
      </Pressable>
    );
  }

  return content;
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: radius.xl,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  default: {},
  primary: {
    backgroundColor: colors.primaryMuted,
    borderColor: colors.primary,
  },
  dark: {
    backgroundColor: colors.secondary,
    borderColor: colors.secondary,
  },
  pressed: {
    opacity: 0.92,
  },
});
