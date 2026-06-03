import { StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { useAuth } from '@/contexts/AuthContext';
import { SettingsRow } from '@/components/SettingsRow';
import { Card } from '@/components/ui/Card';
import { ScreenContainer } from '@/components/ui/ScreenContainer';
import { ScreenHeader } from '@/components/ui/ScreenHeader';
import { settingsSections } from '@/data/mock';
import { colors, spacing, typography } from '@/theme';

function getInitials(name: string): string {
  return name
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);
}

function formatMemberSince(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
  } catch {
    return 'Recently';
  }
}

export default function SettingsScreen() {
  const { user } = useAuth();

  const displayName = user?.full_name ?? 'Guest';
  const displayEmail = user?.email ?? '';
  const displayInitials = user?.full_name ? getInitials(user.full_name) : '?';
  const displayMemberSince = user?.created_at ? formatMemberSince(user.created_at) : '';

  // ✅ FIXED: Inject real user name into the Profile row dynamically
  const resolvedSections = settingsSections.map((section) => ({
    ...section,
    items: section.items.map((item) =>
      item.id === 'profile'
        ? { ...item, value: displayName }
        : item
    ),
  }));

  return (
    <ScreenContainer>
      <ScreenHeader title="Settings" />

      <Card style={styles.profileCard}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>{displayInitials}</Text>
        </View>
        <View style={styles.profileInfo}>
          <Text style={styles.profileName}>{displayName}</Text>
          <Text style={styles.profileEmail}>{displayEmail}</Text>
          {displayMemberSince ? (
            <Text style={styles.member}>Member since {displayMemberSince}</Text>
          ) : null}
        </View>
        <Ionicons name="chevron-forward" size={20} color={colors.textLight} />
      </Card>

      {resolvedSections.map((section) => (
        <View key={section.title} style={styles.section}>
          <Text style={styles.sectionTitle}>{section.title}</Text>
          <Card style={styles.sectionCard}>
            {section.items.map((item, index) => (
              <SettingsRow
                key={item.id}
                label={item.label}
                icon={item.icon}
                value={item.value}
                isLast={index === section.items.length - 1}
                onPress={() => {}}
              />
            ))}
          </Card>
        </View>
      ))}

      <View style={styles.footer}>
        <Text style={styles.footerText}>Secure Verify · v1.0.0</Text>
        <Text style={styles.footerSub}>Built with trust and care 💚</Text>
      </View>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  profileCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    marginBottom: spacing.xl,
    backgroundColor: colors.primaryMuted,
    borderColor: colors.primary,
  },
  avatar: {
    width: 56,
    height: 56,
    borderRadius: 18,
    backgroundColor: colors.secondary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: colors.primary,
    fontSize: 18,
    fontWeight: '700',
  },
  profileInfo: {
    flex: 1,
    gap: 2,
  },
  profileName: {
    ...typography.heading,
    fontSize: 18,
  },
  profileEmail: {
    ...typography.bodySmall,
  },
  member: {
    ...typography.caption,
    marginTop: 2,
  },
  section: {
    marginBottom: spacing.lg,
  },
  sectionTitle: {
    ...typography.label,
    color: colors.textMuted,
    marginBottom: spacing.sm,
    marginLeft: spacing.xs,
  },
  sectionCard: {
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.md,
  },
  footer: {
    alignItems: 'center',
    paddingVertical: spacing.xl,
    gap: spacing.xs,
  },
  footerText: {
    ...typography.caption,
    fontWeight: '600',
  },
  footerSub: {
    ...typography.caption,
  },
});