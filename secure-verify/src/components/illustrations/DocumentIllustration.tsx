import { StyleSheet, Text, View } from 'react-native';

import { colors, radius } from '@/theme';

export function DocumentIllustration() {
  return (
    <View style={styles.container}>
      <View style={styles.doc}>
        <View style={styles.corner} />
        <View style={styles.photo} />
        <View style={styles.line} />
        <View style={styles.line} />
        <View style={[styles.line, styles.lineAccent]} />
      </View>
      <View style={styles.upload}>
        <Text style={styles.arrow}>↑</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    height: 140,
    alignItems: 'center',
    justifyContent: 'center',
  },
  doc: {
    width: 120,
    height: 90,
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    borderWidth: 2,
    borderColor: colors.secondary,
    padding: 12,
    flexDirection: 'row',
    gap: 10,
  },
  corner: {
    position: 'absolute',
    top: 0,
    right: 0,
    width: 20,
    height: 20,
    backgroundColor: colors.primary,
    borderBottomLeftRadius: 8,
  },
  photo: {
    width: 36,
    height: 44,
    backgroundColor: colors.primaryMuted,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: colors.primary,
  },
  line: {
    flex: 1,
    height: 8,
    backgroundColor: colors.border,
    borderRadius: 4,
    marginTop: 8,
  },
  lineAccent: {
    backgroundColor: colors.primary,
    marginTop: 12,
  },
  upload: {
    position: 'absolute',
    bottom: 10,
    right: '30%',
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.secondary,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 3,
    borderColor: colors.primary,
  },
  arrow: {
    color: colors.primary,
    fontSize: 22,
    fontWeight: '700',
  },
});
