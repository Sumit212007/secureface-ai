import { useMemo, useState, useCallback } from 'react';
import { ActivityIndicator, StyleSheet, Text, View, FlatList, RefreshControl } from 'react-native';
import { useFocusEffect } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

import { useAuth } from '@/contexts/AuthContext';
import * as api from '@/services/api';
import { FilterChips } from '@/components/ui/FilterChips';
import { Input } from '@/components/ui/Input';
import { ScreenContainer } from '@/components/ui/ScreenContainer';
import { ScreenHeader } from '@/components/ui/ScreenHeader';
import { Card } from '@/components/ui/Card';
import { HISTORY_FILTERS } from '@/constants/app';
import { colors, spacing, typography } from '@/theme';

interface VerificationRecord {
  id: number;
  user_id: number;
  verification_id: string;
  decision: string;
  similarity: number;
  liveness_score: number;
  timestamp: string;
}

// ✅ FIXED Bug 1: Map display label → backend decision value
const FILTER_TO_DECISION: Record<string, string | null> = {
  All: null,
  Approved: 'ALLOW',
  Pending: 'PENDING',
  Rejected: 'DENY',
};

export default function HistoryScreen() {
  const { user } = useAuth();
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<string>('All');
  const [history, setHistory] = useState<VerificationRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useFocusEffect(
    useCallback(() => {
      loadHistory();
    }, [user?.id])
  );

  const loadHistory = async () => {
    if (!user?.id) return;
    try {
      setLoading(true);
      const response = await api.getHistory(user.id);
      if (response.success) {
        setHistory(response.history);
      }
    } catch (error) {
      console.error('Failed to load history:', error);
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadHistory();
    setRefreshing(false);
  };

  const filtered = useMemo(() => {
    const decisionFilter = FILTER_TO_DECISION[filter]; // null means "All"
    return history.filter((item) => {
      const matchesSearch =
        search.trim() === '' ||
        item.verification_id.toLowerCase().includes(search.toLowerCase());

      // ✅ FIXED Bug 1: Compare against mapped backend value, not display label
      const matchesFilter =
        decisionFilter === null ||
        item.decision === decisionFilter;

      return matchesSearch && matchesFilter;
    });
  }, [search, filter, history]);

  const getStatusColor = (decision: string) => {
    return decision === 'ALLOW' ? '#d1fae5' : '#fee2e2';
  };

  const getStatusTextColor = (decision: string) => {
    return decision === 'ALLOW' ? '#047857' : '#dc2626';
  };

  const renderItem = ({ item }: { item: VerificationRecord }) => (
    <Card style={styles.card}>
      <View style={styles.cardContent}>
        <View style={styles.cardMain}>
          <View style={styles.idRow}>
            <Text style={styles.id}>{item.verification_id}</Text>
            <View style={[styles.badge, { backgroundColor: getStatusColor(item.decision) }]}>
              <Text style={[styles.badgeText, { color: getStatusTextColor(item.decision) }]}>
                {item.decision === 'ALLOW' ? 'Approved' : 'Rejected'}
              </Text>
            </View>
          </View>
          <Text style={styles.date}>
            {new Date(item.timestamp).toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric',
              year: 'numeric',
              hour: '2-digit',
              minute: '2-digit',
            })}
          </Text>
        </View>

        <View style={styles.stats}>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Similarity</Text>
            <Text style={styles.statValue}>{(item.similarity * 100).toFixed(0)}%</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Liveness</Text>
            <Text style={styles.statValue}>{(item.liveness_score * 100).toFixed(0)}%</Text>
          </View>
        </View>
      </View>
    </Card>
  );

  if (loading && history.length === 0) {
    return (
      <ScreenContainer>
        <ScreenHeader title="Verification History" />
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      </ScreenContainer>
    );
  }

  return (
    <ScreenContainer>
      <ScreenHeader title="Verification History" />

      <Text style={styles.subtitle}>{filtered.length} verifications found</Text>

      <View style={styles.searchWrap}>
        <Input
          label="Search"
          placeholder="Search by verification ID..."
          value={search}
          onChangeText={setSearch}
        />
      </View>

      {/* ✅ FIXED Bug 2: Fixed height container stops ScrollView from resizing */}
      <View style={styles.filterRow}>
        <FilterChips
          options={HISTORY_FILTERS}
          selected={filter}
          onSelect={setFilter}
        />
      </View>

      {filtered.length === 0 ? (
        <View style={styles.empty}>
          <Ionicons name="search-outline" size={48} color={colors.textLight} />
          <Text style={styles.emptyTitle}>No results</Text>
          <Text style={styles.emptyText}>Try adjusting your search or filters</Text>
        </View>
      ) : (
        <FlatList
          data={filtered}
          renderItem={renderItem}
          keyExtractor={(item) => item.id.toString()}
          scrollEnabled={false}
          contentContainerStyle={styles.list}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
          }
        />
      )}
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  subtitle: {
    ...typography.bodySmall,
    marginBottom: spacing.md,
  },
  searchWrap: {
    marginBottom: spacing.sm,
  },
  // ✅ FIXED Bug 2: Fixed height prevents layout shift on filter change
  filterRow: {
    height: 52,
    marginBottom: spacing.sm,
  },
  list: {
    gap: spacing.md,
    marginTop: spacing.md,
  },
  empty: {
    alignItems: 'center',
    paddingVertical: spacing.xxl,
    gap: spacing.sm,
  },
  emptyTitle: {
    ...typography.heading,
  },
  emptyText: {
    ...typography.bodySmall,
    color: colors.textLight,
  },
  card: {
    padding: spacing.md,
  },
  cardContent: {
    gap: spacing.md,
  },
  cardMain: {
    gap: spacing.xs,
  },
  idRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  id: {
    ...typography.heading,
    fontSize: 14,
    flex: 1,
  },
  badge: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: 16,
    marginLeft: spacing.sm,
  },
  badgeText: {
    fontSize: 12,
    fontWeight: '600',
  },
  date: {
    ...typography.bodySmall,
    color: colors.textLight,
  },
  stats: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statLabel: {
    ...typography.bodySmall,
    color: colors.textLight,
    marginBottom: spacing.xs,
  },
  statValue: {
    ...typography.heading,
    fontSize: 16,
  },
  statDivider: {
    width: 1,
    height: 30,
    backgroundColor: colors.border,
    marginHorizontal: spacing.sm,
  },
});