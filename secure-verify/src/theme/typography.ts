import { Platform, TextStyle } from 'react-native';

import { colors } from './colors';

const fontFamily = Platform.select({
  ios: 'System',
  android: 'Roboto',
  default: 'System',
});

export const typography = {
  hero: {
    fontFamily,
    fontSize: 32,
    fontWeight: '700',
    lineHeight: 40,
    color: colors.text,
    letterSpacing: -0.5,
  } as TextStyle,
  title: {
    fontFamily,
    fontSize: 26,
    fontWeight: '700',
    lineHeight: 32,
    color: colors.text,
    letterSpacing: -0.3,
  } as TextStyle,
  heading: {
    fontFamily,
    fontSize: 20,
    fontWeight: '600',
    lineHeight: 26,
    color: colors.text,
  } as TextStyle,
  subtitle: {
    fontFamily,
    fontSize: 16,
    fontWeight: '400',
    lineHeight: 24,
    color: colors.textMuted,
  } as TextStyle,
  body: {
    fontFamily,
    fontSize: 16,
    fontWeight: '400',
    lineHeight: 22,
    color: colors.text,
  } as TextStyle,
  bodySmall: {
    fontFamily,
    fontSize: 14,
    fontWeight: '400',
    lineHeight: 20,
    color: colors.textMuted,
  } as TextStyle,
  label: {
    fontFamily,
    fontSize: 13,
    fontWeight: '600',
    lineHeight: 18,
    color: colors.text,
    letterSpacing: 0.2,
  } as TextStyle,
  caption: {
    fontFamily,
    fontSize: 12,
    fontWeight: '500',
    lineHeight: 16,
    color: colors.textLight,
  } as TextStyle,
  button: {
    fontFamily,
    fontSize: 17,
    fontWeight: '600',
    lineHeight: 22,
    letterSpacing: 0.1,
  } as TextStyle,
};
