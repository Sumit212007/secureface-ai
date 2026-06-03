import { ActivityIndicator, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { PickedImage } from '@/utils/verificationMedia';
import { colors, radius, spacing, typography } from '@/theme';

interface DocumentUploadStepProps {
  document: PickedImage | null;
  isPicking: boolean;
  error: string | null;
  onPick: () => void;
}

export function DocumentUploadStep({
  document,
  isPicking,
  error,
  onPick,
}: DocumentUploadStepProps) {
  const uploaded = document !== null;

  return (
    <View style={styles.stepContent}>
      <Text style={styles.stepTitle}>Upload your document</Text>
      <Text style={styles.stepDesc}>
        Tap below to choose a clear photo of your passport, license, or national ID.
      </Text>

      <Pressable
        onPress={onPick}
        disabled={isPicking}
        style={({ pressed }) => [
          styles.uploadZone,
          uploaded && styles.uploadZoneSuccess,
          pressed && !isPicking && styles.uploadPressed,
        ]}>
        {isPicking ? (
          <View style={styles.loadingWrap}>
            <ActivityIndicator size="large" color={colors.secondary} />
            <Text style={styles.uploadText}>Opening gallery…</Text>
          </View>
        ) : uploaded ? (
          <>
            <Image source={{ uri: document.uri }} style={styles.preview} resizeMode="cover" />
            {document.fileName ? (
              <Text style={styles.fileName} numberOfLines={1}>
                {document.fileName}
              </Text>
            ) : null}
            <View style={styles.successRow}>
              <Ionicons name="checkmark-circle" size={22} color={colors.success} />
              <Text style={styles.successText}>Document uploaded successfully</Text>
            </View>
            <Text style={styles.changeHint}>Tap to change image</Text>
          </>
        ) : (
          <>
            <Ionicons name="cloud-upload-outline" size={32} color={colors.secondary} />
            <Text style={styles.uploadText}>Tap to upload from gallery</Text>
            <Text style={styles.uploadHint}>JPG, JPEG, PNG · Max 10MB</Text>
          </>
        )}
      </Pressable>

      {error ? (
        <View style={styles.errorBox}>
          <Ionicons name="alert-circle" size={20} color={colors.error} />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : null}

      <View style={styles.docTypes}>
        {['Passport', 'License', 'National ID'].map((type) => (
          <View key={type} style={styles.docChip}>
            <Text style={styles.docChipText}>{type}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  stepContent: {
    alignItems: 'center',
    gap: spacing.md,
  },
  stepTitle: {
    ...typography.heading,
    textAlign: 'center',
  },
  stepDesc: {
    ...typography.bodySmall,
    textAlign: 'center',
    paddingHorizontal: spacing.sm,
  },
  uploadZone: {
    alignSelf: 'stretch',
    borderWidth: 2,
    borderStyle: 'dashed',
    borderColor: colors.primary,
    borderRadius: radius.xl,
    backgroundColor: colors.primaryMuted,
    padding: spacing.lg,
    alignItems: 'center',
    gap: spacing.sm,
    minHeight: 200,
    overflow: 'hidden',
  },
  uploadZoneSuccess: {
    borderStyle: 'solid',
    borderColor: colors.success,
    backgroundColor: '#E8F9EE',
  },
  uploadPressed: {
    opacity: 0.9,
  },
  loadingWrap: {
    paddingVertical: spacing.xl,
    alignItems: 'center',
    gap: spacing.md,
  },
  preview: {
    width: '100%',
    height: 160,
    borderRadius: radius.lg,
    backgroundColor: colors.border,
  },
  fileName: {
    ...typography.caption,
    fontWeight: '600',
    alignSelf: 'stretch',
    textAlign: 'center',
  },
  successRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.xs,
  },
  successText: {
    ...typography.bodySmall,
    color: colors.success,
    fontWeight: '600',
  },
  changeHint: {
    ...typography.caption,
    color: colors.textMuted,
  },
  uploadText: {
    ...typography.label,
  },
  uploadHint: {
    ...typography.caption,
  },
  errorBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    alignSelf: 'stretch',
    padding: spacing.md,
    backgroundColor: '#FFEBE9',
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.error,
  },
  errorText: {
    ...typography.bodySmall,
    color: colors.error,
    flex: 1,
  },
  docTypes: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    justifyContent: 'center',
  },
  docChip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    backgroundColor: colors.background,
    borderRadius: radius.full,
    borderWidth: 1,
    borderColor: colors.border,
  },
  docChipText: {
    ...typography.caption,
    fontWeight: '600',
  },
});
