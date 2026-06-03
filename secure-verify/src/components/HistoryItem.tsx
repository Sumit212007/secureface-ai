import { StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { Card } from '@/components/ui/Card';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { VerificationRecord } from '@/data/mock';
import { colors, spacing, typography } from '@/theme';

interface HistoryItemProps {
  item: VerificationRecord;
  onPress?: () => void;
}

export function HistoryItem({ item, onPress }: HistoryItemProps) {
  return (
    <Card onPress={onPress} style={styles.card}>
      <View style={styles.row}>
        <View style={styles.icon}>
          <Ionicons name="document-text-outline" size={22} color={colors.secondary} />
        </View>
        <View style={styles.content}>
          <Text style={styles.title}>{item.title}</Text>
          <Text style={styles.meta}>
            {item.documentType} · {item.date}
          </Text>
          <Text style={styles.id}>{item.id}</Text>
        </View>
        <StatusBadge status={item.status} />
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: spacing.md,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.md,
  },
  icon: {
    width: 44,
    height: 44,
    borderRadius: 14,
    backgroundColor: colors.primaryMuted,
    alignItems: 'center',
    justifyContent: 'center',
  },
  content: {
    flex: 1,
    gap: 2,
  },
  title: {
    ...typography.label,
    fontSize: 15,
  },
  meta: {
    ...typography.caption,
  },
  id: {
    ...typography.caption,
    color: colors.textLight,
    marginTop: 2,
  },
});
