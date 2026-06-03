import { StyleSheet, Text, View } from 'react-native';

import { colors, radius } from '@/theme';

export function SecureIllustration() {
  return (
    <View style={styles.container}>
      <View style={styles.ring} />
      <View style={styles.phone}>
        <View style={styles.notch} />
        <View style={styles.screen}>
          <Text style={styles.lock}>🔒</Text>
          <View style={styles.bar} />
          <View style={[styles.bar, styles.barShort]} />
        </View>
      </View>
      <View style={styles.checkBadge}>
        <Text style={styles.check}>✓</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: 160,
    height: 180,
    alignItems: 'center',
    justifyContent: 'center',
  },
  ring: {
    position: 'absolute',
    width: 140,
    height: 140,
    borderRadius: 70,
    borderWidth: 3,
    borderColor: colors.primary,
    borderStyle: 'dashed',
  },
  phone: {
    width: 90,
    height: 150,
    backgroundColor: colors.secondary,
    borderRadius: 20,
    padding: 8,
    alignItems: 'center',
  },
  notch: {
    width: 36,
    height: 6,
    backgroundColor: colors.textMuted,
    borderRadius: 3,
    marginBottom: 12,
  },
  screen: {
    flex: 1,
    width: '100%',
    backgroundColor: colors.primaryMuted,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    padding: 12,
  },
  lock: {
    fontSize: 32,
  },
  bar: {
    width: '80%',
    height: 8,
    backgroundColor: colors.primary,
    borderRadius: 4,
  },
  barShort: {
    width: '50%',
    backgroundColor: colors.border,
  },
  checkBadge: {
    position: 'absolute',
    bottom: 10,
    right: 10,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.success,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 3,
    borderColor: colors.surface,
  },
  check: {
    color: colors.surface,
    fontSize: 20,
    fontWeight: '700',
  },
});
