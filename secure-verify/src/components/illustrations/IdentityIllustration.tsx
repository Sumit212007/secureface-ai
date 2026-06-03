import { StyleSheet, Text, View } from 'react-native';

import { colors, radius } from '@/theme';

export function IdentityIllustration() {
  return (
    <View style={styles.container}>
      <View style={styles.bgCircle} />
      <View style={styles.card}>
        <View style={styles.avatar}>
          <Text style={styles.face}>😊</Text>
        </View>
        <View style={styles.lines}>
          <View style={[styles.line, styles.lineLong]} />
          <View style={styles.line} />
          <View style={styles.line} />
        </View>
      </View>
      <View style={styles.shield}>
        <Text style={styles.shieldIcon}>🛡️</Text>
      </View>
      <View style={styles.sparkle}>
        <Text>✨</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: 220,
    height: 200,
    alignSelf: 'center',
    alignItems: 'center',
    justifyContent: 'center',
  },
  bgCircle: {
    position: 'absolute',
    width: 180,
    height: 180,
    borderRadius: 90,
    backgroundColor: colors.primaryMuted,
  },
  card: {
    width: 140,
    height: 100,
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    borderWidth: 2,
    borderColor: colors.secondary,
    flexDirection: 'row',
    padding: 12,
    gap: 10,
    transform: [{ rotate: '-4deg' }],
  },
  avatar: {
    width: 48,
    height: 56,
    backgroundColor: colors.primary,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  face: {
    fontSize: 28,
  },
  lines: {
    flex: 1,
    justifyContent: 'center',
    gap: 8,
  },
  line: {
    height: 8,
    backgroundColor: colors.border,
    borderRadius: 4,
    width: '80%',
  },
  lineLong: {
    width: '100%',
    backgroundColor: colors.primary,
  },
  shield: {
    position: 'absolute',
    right: 20,
    bottom: 30,
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: colors.secondary,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 3,
    borderColor: colors.primary,
  },
  shieldIcon: {
    fontSize: 24,
  },
  sparkle: {
    position: 'absolute',
    top: 20,
    right: 50,
    fontSize: 20,
  },
});
