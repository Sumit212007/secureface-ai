import { StyleSheet, Text, View } from 'react-native';

import { VerificationStatus } from '@/data/mock';
import { colors, radius, typography } from '@/theme';

const config: Record<VerificationStatus, { label: string; bg: string; text: string }> = {
  approved: { label: 'Approved', bg: '#E8F9EE', text: colors.success },
  pending: { label: 'Pending', bg: '#FFF4E5', text: colors.warning },
  rejected: { label: 'Rejected', bg: '#FFEBE9', text: colors.error },
};

interface StatusBadgeProps {
  status: VerificationStatus;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const { label, bg, text } = config[status];

  return (
    <View style={[styles.badge, { backgroundColor: bg }]}>
      <Text style={[styles.text, { color: text }]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: radius.full,
  },
  text: {
    ...typography.caption,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.4,
  },
});
