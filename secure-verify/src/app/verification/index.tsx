import { useCallback, useEffect, useRef, useState } from 'react';
import { InteractionManager, Platform, StyleSheet, Text, View } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

import { useAuth } from '@/contexts/AuthContext';
import { DocumentUploadStep } from '@/components/verification/DocumentUploadStep';
import { FaceVerificationStep } from '@/components/verification/FaceVerificationStep';
import { SelfieCameraModal } from '@/components/verification/SelfieCameraModal';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { ScreenContainer } from '@/components/ui/ScreenContainer';
import { ScreenHeader } from '@/components/ui/ScreenHeader';
import { StepProgress } from '@/components/ui/StepProgress';
import { Routes } from '@/constants/routes';
import { generateVerificationId } from '@/data/mock';
import { useRecoverPendingSelfie } from '@/hooks/useRecoverPendingSelfie';
import {
  PickedImage,
  captureSelfieWithSystemCamera,
  pickDocumentImage,
  recoverPendingPickerResult,
  useInAppSelfieCamera,
} from '@/utils/verificationMedia';
import { colors, spacing, typography } from '@/theme';

const LOG = '[VerificationScreen]';
const TOTAL_STEPS = 2;
const VERIFY_URL = 'http://192.168.1.108:5000/verify';

export default function VerificationScreen() {
  const router = useRouter();
  const { offline } = useLocalSearchParams<{ offline?: string }>();
  const { user } = useAuth();
  const [step, setStep] = useState(1);
  const [navigating, setNavigating] = useState(false);

  const [document, setDocument] = useState<PickedImage | null>(null);
  const [documentPicking, setDocumentPicking] = useState(false);
  const [documentError, setDocumentError] = useState<string | null>(null);

  const [selfie, setSelfie] = useState<PickedImage | null>(null);
  const [faceCapturing, setFaceCapturing] = useState(false);
  const [faceVerifying, setFaceVerifying] = useState(false);
  const [completedCheckCount, setCompletedCheckCount] = useState(0);
  const [faceComplete, setFaceComplete] = useState(false);
  const [faceError, setFaceError] = useState<string | null>(null);
  const [selfieCameraOpen, setSelfieCameraOpen] = useState(false);
  const [verificationResult, setVerificationResult] = useState<any>(null);

  const checkTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const coldStartRecoveryDone = useRef(false);

  const isOffline = offline === '1';

  const clearCheckTimers = useCallback(() => {
    checkTimersRef.current.forEach(clearTimeout);
    checkTimersRef.current = [];
  }, []);

  useEffect(() => () => clearCheckTimers(), [clearCheckTimers]);

  const verifyFace = useCallback(async (image: PickedImage) => {
    console.log(LOG, 'verifyFace: start', { uri: image.uri.slice(0, 80) });
    try {
      setFaceVerifying(true);
      setCompletedCheckCount(0);

      const result = await new Promise<any>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', VERIFY_URL);

        xhr.onload = () => {
          console.log(LOG, 'verifyFace: XHR loaded with status', xhr.status);
          try {
            const responseData = JSON.parse(xhr.responseText);
            resolve(responseData);
          } catch (e) {
            reject(new Error(`Failed to parse response: ${xhr.responseText}`));
          }
        };

        xhr.onerror = (e) => {
          console.error(LOG, 'verifyFace: XHR error event', e);
          reject(new Error('Network request failed'));
        };

        const formData = new FormData();
        formData.append('image', {
          uri: image.uri,
          name: image.fileName || 'selfie.jpg',
          type: 'image/jpeg',
        } as any);

        // Add user_id to save verification to history
        if (user?.id) {
          formData.append('user_id', user.id.toString());
        }

        xhr.send(formData);
      });

      console.log(LOG, 'verifyFace: response', result);

      setCompletedCheckCount(4);

      if (result.success && result.decision === 'ALLOW') {
        setFaceComplete(true);
        setFaceError(null);
        setVerificationResult(result);
        return true;
      }

      setFaceComplete(false);
      setFaceError(result.message || 'Face verification failed');
      setVerificationResult(result);
      return false;
    } catch (error) {
      console.error(LOG, 'verifyFace: error', error);
      setFaceComplete(false);
      setFaceError('Cannot connect to SecureEdge AI backend');
      return false;
    } finally {
      setFaceVerifying(false);
      console.log(LOG, 'verifyFace: done');
    }
  }, [user?.id]);

  const applyCapturedSelfie = useCallback(
    (image: PickedImage) => {
      console.log(LOG, 'applyCapturedSelfie', { uri: image.uri.slice(0, 80) });
      setStep(2);
      setSelfie(image);
      setFaceComplete(false);
      setCompletedCheckCount(0);
      void verifyFace(image);
    },
    [verifyFace]
  );

  // Fallback if a previous build used launchCameraAsync and MainActivity was destroyed (same process).
  useRecoverPendingSelfie(step === 2 && !selfie, applyCapturedSelfie);

  // After full process kill + reload, jump back to face step if picker stashed a pending result.
  useEffect(() => {
    if (coldStartRecoveryDone.current) {
      return;
    }
    coldStartRecoveryDone.current = true;

    void (async () => {
      console.log(LOG, 'coldStart: checking pending picker result');
      const { image, error } = await recoverPendingPickerResult();
      if (error) {
        setFaceError(error);
        setStep(2);
        return;
      }
      if (image) {
        console.log(LOG, 'coldStart: recovered pending selfie after reload');
        applyCapturedSelfie(image);
      }
    })();
  }, [applyCapturedSelfie]);

  const handlePickDocument = async () => {
    setDocumentError(null);
    setDocumentPicking(true);
    const { image, error } = await pickDocumentImage();
    setDocumentPicking(false);
    if (error) {
      setDocumentError(error);
      return;
    }
    if (image) {
      setDocument(image);
    }
  };

  const handleCaptureSelfie = async () => {
    console.log(LOG, 'handleCaptureSelfie: start', {
      platform: Platform.OS,
      useInAppSelfieCamera,
    });
    setFaceError(null);

    if (useInAppSelfieCamera) {
      console.log(LOG, 'handleCaptureSelfie: opening in-app camera modal');
      setSelfieCameraOpen(true);
      return;
    }

    setFaceCapturing(true);
    let image: PickedImage | null = null;
    let error: string | null = null;

    try {
      const result = await captureSelfieWithSystemCamera();
      image = result.image;
      error = result.error;
      console.log(LOG, 'handleCaptureSelfie: system camera returned', { hasImage: !!image, error });
    } catch (err) {
      console.error(LOG, 'handleCaptureSelfie: unexpected throw', err);
      error = 'Could not capture a photo. Please try again.';
    } finally {
      setFaceCapturing(false);
    }

    if (error) {
      setFaceError(error);
      return;
    }
    if (!image) {
      return;
    }

    await new Promise<void>((resolve) => {
      InteractionManager.runAfterInteractions(() => {
        applyCapturedSelfie(image!);
        resolve();
      });
    });
  };

  const handleInAppSelfieCaptured = (image: PickedImage) => {
    console.log(LOG, 'handleInAppSelfieCaptured');
    setSelfieCameraOpen(false);
    applyCapturedSelfie(image);
  };

  const handleRetrySelfie = () => {
    clearCheckTimers();
    setSelfie(null);
    setFaceVerifying(false);
    setCompletedCheckCount(0);
    setFaceComplete(false);
    setFaceError(null);
  };

  const canContinue =
    step === 1 ? document !== null : faceComplete && !faceVerifying && !faceCapturing;

  const goNext = () => {
    if (!canContinue) return;

    if (step < TOTAL_STEPS) {
      setStep((s) => s + 1);
      return;
    }

    setNavigating(true);
    const id = verificationResult?.verification_id || generateVerificationId();
    const timestamp = new Date().toISOString();
    setTimeout(() => {
      setNavigating(false);
      router.replace({
        pathname: Routes.verificationSuccess,
        params: { 
          id, 
          timestamp,
          similarity: verificationResult?.similarity?.toString(),
          livenessScore: verificationResult?.liveness_score?.toString(),
        },
      });
    }, 600);
  };

  return (
    <ScreenContainer>
      <ScreenHeader title="Verification" />

      {isOffline && (
        <View style={styles.offlineBanner}>
          <Ionicons name="cloud-offline-outline" size={18} color={colors.secondary} />
          <Text style={styles.offlineText}>Offline mode — data syncs when connected</Text>
        </View>
      )}

      <StepProgress currentStep={step} />

      <Card style={styles.stepCard}>
        {step === 1 ? (
          <DocumentUploadStep
            document={document}
            isPicking={documentPicking}
            error={documentError}
            onPick={handlePickDocument}
          />
        ) : (
          <FaceVerificationStep
            selfie={selfie}
            isCapturing={faceCapturing || selfieCameraOpen}
            isVerifying={faceVerifying}
            completedCheckCount={completedCheckCount}
            verificationComplete={faceComplete}
            error={faceError}
            onCapture={handleCaptureSelfie}
            onRetry={handleRetrySelfie}
          />
        )}
      </Card>

      <SelfieCameraModal
        visible={selfieCameraOpen}
        onClose={() => setSelfieCameraOpen(false)}
        onCaptured={handleInAppSelfieCaptured}
      />

      <View style={styles.actions}>
        {step > 1 && (
          <Button
            label="Back"
            variant="outline"
            onPress={() => setStep((s) => s - 1)}
            disabled={navigating || faceVerifying}
            style={styles.backBtn}
          />
        )}
        <Button
          label={step === TOTAL_STEPS ? 'Complete Verification' : 'Continue'}
          onPress={goNext}
          loading={navigating}
          disabled={!canContinue}
          style={styles.continueBtn}
        />
      </View>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  offlineBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: colors.primaryMuted,
    padding: spacing.md,
    borderRadius: 12,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.primary,
  },
  offlineText: {
    ...typography.bodySmall,
    flex: 1,
    fontWeight: '500',
  },
  stepCard: {
    marginBottom: spacing.lg,
  },
  actions: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  backBtn: {
    flex: 1,
  },
  continueBtn: {
    flex: 2,
  },
});
