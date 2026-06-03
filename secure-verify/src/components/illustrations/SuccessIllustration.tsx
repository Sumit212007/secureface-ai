import { StyleSheet, Text, View } from 'react-native';

import { colors, radius } from '@/theme';

export function SuccessIllustration() {
  return (
    <View style={styles.container}>
      <View style={styles.outerRing} />
      <View style={styles.innerCircle}>
        <Text style={styles.emoji}>🎉</Text>
      </View>
      <View style={[styles.confetti, styles.c1]}>
        <Text>🟢</Text>
      </View>
      <View style={[styles.confetti, styles.c2]}>
        <Text>⭐</Text>
      </View>
      <View style={[styles.confetti, styles.c3]}>
        <Text>💚</Text>
      </View>
      <View style={styles.badge}>
        <Text style={styles.badgeText}>✓</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: 240,
    height: 240,
    alignSelf: 'center',
    alignItems: 'center',
    justifyContent: 'center',
  },
  outerRing: {
    position: 'absolute',
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: colors.primaryMuted,
  },
  innerCircle: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 4,
    borderColor: colors.secondary,
  },
  emoji: {
    fontSize: 56,
  },
  confetti: {
    position: 'absolute',
    fontSize: 22,
  },
  c1: { top: 30, left: 30 },
  c2: { top: 50, right: 25 },
  c3: { bottom: 50, left: 40 },
  badge: {
    position: 'absolute',
    bottom: 35,
    right: 45,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: colors.success,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 4,
    borderColor: colors.surface,
  },
  badgeText: {
    fontSize: 28,
    fontWeight: '800',
    color: colors.surface,
  },
});
