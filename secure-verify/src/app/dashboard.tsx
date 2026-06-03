import { useCallback, useEffect, useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, View } from 'react-native';
import { useRouter, useFocusEffect } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

import { useAuth } from '@/contexts/AuthContext';
import * as api from '@/services/api';
import { AppLogo } from '@/components/AppLogo';
import { DashboardCard } from '@/components/DashboardCard';
import { HistoryItem } from '@/components/HistoryItem';
import { Card } from '@/components/ui/Card';
import { ScreenContainer } from '@/components/ui/ScreenContainer';
import { StatCard } from '@/components/ui/StatCard';
import { Routes } from '@/constants/routes';
import { mockRecentVerifications } from '@/data/mock';
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

export default function DashboardScreen() {
  const router = useRouter();
  const { user, logout } = useAuth();
  const [stats, setStats] = useState({
    totalVerifications: 0,
    approved: 0,
    successRate: 0,
  });
  const [recentVerifications, setRecentVerifications] = useState<VerificationRecord[]>([]);

  // Load verification history when screen comes into focus
  useFocusEffect(
    useCallback(() => {
      if (user?.id) {
        loadVerificationHistory();
      }
    }, [user?.id])
  );

  const loadVerificationHistory = async () => {
    if (!user?.id) return;

    try {
      const response = await api.getHistory(user.id);
      if (response.success && response.history.length > 0) {
        setRecentVerifications(response.history.slice(0, 2));
        
        // Calculate stats
        const total = response.history.length;
        const approved = response.history.filter(
          (v: VerificationRecord) => v.decision === 'ALLOW'
        ).length;
        const successRate = total > 0 ? Math.round((approved / total) * 100) : 0;

        setStats({
          totalVerifications: total,
          approved,
          successRate,
        });
      }
    } catch (error) {
      console.error('Failed to load verification history:', error);
    }
  };

  const handleLogout = () => {
    Alert.alert('Logout', 'Are you sure you want to log out?', [
      { text: 'Cancel', onPress: () => {}, style: 'cancel' },
      {
        text: 'Logout',
        onPress: async () => {
          await logout();
          router.replace(Routes.login);
        },
        style: 'destructive',
      },
    ]);
  };

  const handleSettings = () => {
    router.push(Routes.settings);
  };

  const firstName = user?.full_name.split(' ')[0] || 'User';

  return (
    <ScreenContainer>
      <View style={styles.header}>
        <AppLogo size="sm" showText={false} />
        <View style={styles.headerRight}>
          <Pressable onPress={handleSettings} style={styles.iconButton}>
            <Ionicons name="settings-outline" size={24} color={colors.text} />
          </Pressable>
          <Pressable onPress={handleLogout} style={styles.iconButton}>
            <Ionicons name="log-out-outline" size={24} color={colors.danger} />
          </Pressable>
        </View>
      </View>

      <View style={styles.welcome}>
        <Text style={styles.greeting}>Welcome back,</Text>
        <Text style={styles.name}>{firstName} 👋</Text>
        <Text style={styles.email}>{user?.email}</Text>
        <Text style={styles.welcomeSub}>Your identity hub is ready</Text>
      </View>

      <View style={styles.stats}>
        <StatCard value={stats.totalVerifications} label="Total" accent />
        <StatCard value={`${stats.successRate}%`} label="Success" />
        <StatCard value={stats.approved} label="Approved" />
      </View>

      <Text style={styles.sectionTitle}>Quick Actions</Text>
      <View style={styles.grid}>
        <View style={styles.gridRow}>
          <View style={styles.gridItem}>
            <DashboardCard
              title="Start Verification"
              subtitle="Begin a new ID check"
              icon="scan-outline"
              variant="primary"
              onPress={() => router.push(Routes.verification)}
            />
          </View>
          <View style={styles.gridItem}>
            <DashboardCard
              title="Verification History"
              subtitle="View past checks"
              icon="time-outline"
              onPress={() => router.push(Routes.history)}
            />
          </View>
        </View>
        <View style={styles.gridRow}>
          <View style={styles.gridItem}>
            <DashboardCard
              title="Offline Mode"
              subtitle="Verify without network"
              icon="cloud-offline-outline"
              onPress={() =>
                router.push({ pathname: Routes.verification, params: { offline: '1' } })
              }
            />
          </View>
          <View style={styles.gridItem}>
            <DashboardCard
              title="Settings"
              subtitle="Account & preferences"
              icon="settings-outline"
              variant="dark"
              onPress={handleSettings}
            />
          </View>
        </View>
      </View>

      <View style={styles.recentHeader}>
        <Text style={styles.sectionTitle}>Recent Verifications</Text>
        <Text style={styles.seeAll} onPress={() => router.push(Routes.history)}>
          See all
        </Text>
      </View>

      <View style={styles.recentList}>
        {recentVerifications.length > 0 ? (
          recentVerifications.map((item) => (
            <Card key={item.id} style={styles.historyCard}>
              <View style={styles.historyContent}>
                <View style={styles.historyMain}>
                  <Text style={styles.historyId}>{item.verification_id}</Text>
                  <Text style={styles.historyDate}>
                    {new Date(item.timestamp).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric',
                    })}
                  </Text>
                </View>
                <View style={styles.historyStats}>
                  <View style={styles.statBadge}>
                    <Text style={styles.statValue}>{(item.similarity * 100).toFixed(0)}%</Text>
                    <Text style={styles.statLabel}>Similarity</Text>
                  </View>
                </View>
                <View
                  style={[
                    styles.decisionBadge,
                    item.decision === 'ALLOW' ? styles.approved : styles.rejected,
                  ]}
                >
                  <Text style={styles.decisionText}>{item.decision}</Text>
                </View>
              </View>
            </Card>
          ))
        ) : (
          <View style={styles.emptyState}>
            <Ionicons name="checkmark-circle-outline" size={48} color={colors.textLight} />
            <Text style={styles.emptyTitle}>No verifications yet</Text>
            <Text style={styles.emptyText}>Start your first verification to see results here</Text>
          </View>
        )}
      </View>

      <Card style={styles.tip}>
        <View style={styles.tipRow}>
          <Ionicons name="bulb-outline" size={22} color={colors.secondary} />
          <View style={styles.tipText}>
            <Text style={styles.tipTitle}>Pro tip</Text>
            <Text style={styles.tipBody}>
              Good lighting helps face verification pass on the first try.
            </Text>
          </View>
        </View>
      </Card>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerRight: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  iconButton: {
    padding: spacing.sm,
  },
  welcome: {
    gap: spacing.xs,
    marginBottom: spacing.md,
  },
  greeting: {
    ...typography.bodySmall,
    color: colors.textLight,
  },
  name: {
    ...typography.title,
  },
  email: {
    ...typography.bodySmall,
    color: colors.textLight,
    marginBottom: spacing.xs,
  },
  welcomeSub: {
    ...typography.subtitle,
  },
  stats: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginBottom: spacing.lg,
  },
  grid: {
    gap: spacing.md,
  },
  gridRow: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  gridItem: {
    flex: 1,
  },
  sectionTitle: {
    ...typography.heading,
    marginVertical: spacing.md,
  },
  recentHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: spacing.lg,
  },
  seeAll: {
    color: colors.secondary,
    fontSize: 14,
    fontWeight: '500',
  },
  recentList: {
    gap: spacing.md,
    marginTop: spacing.md,
  },
  historyCard: {
    padding: spacing.md,
  },
  historyContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  historyMain: {
    flex: 1,
  },
  historyId: {
    ...typography.heading,
    fontSize: 14,
    marginBottom: spacing.xs,
  },
  historyDate: {
    ...typography.bodySmall,
    color: colors.textLight,
  },
  historyStats: {
    alignItems: 'center',
    marginHorizontal: spacing.md,
  },
  statBadge: {
    alignItems: 'center',
  },
  statValue: {
    ...typography.heading,
    fontSize: 14,
  },
  statLabel: {
    ...typography.bodySmall,
    color: colors.textLight,
    fontSize: 11,
  },
  decisionBadge: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: 20,
  },
  approved: {
    backgroundColor: '#d1fae5',
  },
  rejected: {
    backgroundColor: '#fee2e2',
  },
  decisionText: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.text,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: spacing.lg,
    gap: spacing.sm,
  },
  emptyTitle: {
    ...typography.heading,
  },
  emptyText: {
    ...typography.bodySmall,
    color: colors.textLight,
  },
  tip: {
    marginTop: spacing.lg,
    backgroundColor: '#f0fdf4',
    padding: spacing.md,
  },
  tipRow: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  tipText: {
    flex: 1,
  },
  tipTitle: {
    fontWeight: '600',
    color: colors.text,
    marginBottom: spacing.xs,
  },
  tipBody: {
    ...typography.bodySmall,
    color: colors.textLight,
  },
});
