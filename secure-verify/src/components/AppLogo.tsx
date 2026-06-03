import { StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { APP_NAME } from '@/constants/app';
import { colors, radius, spacing, typography } from '@/theme';

interface AppLogoProps {
  size?: 'sm' | 'md' | 'lg';
  showText?: boolean;
}

const sizes = {
  sm: { box: 36, icon: 18, text: typography.bodySmall },
  md: { box: 48, icon: 24, text: typography.heading },
  lg: { box: 64, icon: 32, text: typography.title },
};

export function AppLogo({ size = 'md', showText = true }: AppLogoProps) {
  const s = sizes[size];

  return (
    <View style={styles.row}>
      <View style={[styles.iconBox, { width: s.box, height: s.box, borderRadius: s.box * 0.3 }]}>
        <Ionicons name="shield-checkmark" size={s.icon} color={colors.secondary} />
      </View>
      {showText && (
        <View>
          <Text style={[s.text, styles.name]}>{APP_NAME}</Text>
          {size === 'lg' && <Text style={styles.tagline}>Trust. Verify. Secure.</Text>}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  iconBox: {
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: colors.secondary,
  },
  name: {
    fontWeight: '700',
  },
  tagline: {
    ...typography.caption,
    marginTop: 2,
  },
});
