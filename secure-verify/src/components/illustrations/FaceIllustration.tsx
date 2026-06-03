import { StyleSheet, Text, View } from 'react-native';

import { colors, radius } from '@/theme';

export function FaceIllustration() {
  return (
    <View style={styles.container}>
      <View style={styles.frame}>
        <View style={styles.faceCircle}>
          <Text style={styles.face}>🙂</Text>
        </View>
        <View style={[styles.corner, styles.tl]} />
        <View style={[styles.corner, styles.tr]} />
        <View style={[styles.corner, styles.bl]} />
        <View style={[styles.corner, styles.br]} />
      </View>
      <Text style={styles.hint}>Align your face in the frame</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    gap: 16,
  },
  frame: {
    width: 180,
    height: 180,
    alignItems: 'center',
    justifyContent: 'center',
  },
  faceCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: colors.primaryMuted,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 3,
    borderColor: colors.primary,
  },
  face: {
    fontSize: 56,
  },
  corner: {
    position: 'absolute',
    width: 28,
    height: 28,
    borderColor: colors.secondary,
  },
  tl: { top: 0, left: 0, borderTopWidth: 4, borderLeftWidth: 4, borderTopLeftRadius: 12 },
  tr: { top: 0, right: 0, borderTopWidth: 4, borderRightWidth: 4, borderTopRightRadius: 12 },
  bl: { bottom: 0, left: 0, borderBottomWidth: 4, borderLeftWidth: 4, borderBottomLeftRadius: 12 },
  br: { bottom: 0, right: 0, borderBottomWidth: 4, borderRightWidth: 4, borderBottomRightRadius: 12 },
  hint: {
    fontSize: 14,
    color: colors.textMuted,
  },
});
