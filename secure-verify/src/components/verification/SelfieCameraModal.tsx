import { useCallback, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';

import { Button } from '@/components/ui/Button';
import { PickedImage } from '@/utils/verificationMedia';
import { colors, radius, spacing, typography } from '@/theme';

const LOG = '[SelfieCameraModal]';

interface SelfieCameraModalProps {
  visible: boolean;
  onClose: () => void;
  onCaptured: (image: PickedImage) => void;
}

export function SelfieCameraModal({ visible, onClose, onCaptured }: SelfieCameraModalProps) {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraReady, setCameraReady] = useState(false);
  const [capturing, setCapturing] = useState(false);

  const handleCapture = useCallback(async () => {
    if (!cameraRef.current || !cameraReady || capturing) {
      return;
    }

    setCapturing(true);
    console.log(LOG, 'takePictureAsync: start');

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.85,
        skipProcessing: false,
      });

      console.log(LOG, 'takePictureAsync: success', { uri: photo.uri?.slice(0, 80) });

      onCaptured({
        uri: photo.uri,
        fileName: 'selfie.jpg',
      });
      onClose();
    } catch (err) {
      console.error(LOG, 'takePictureAsync: failed', err);
    } finally {
      setCapturing(false);
    }
  }, [cameraReady, capturing, onCaptured, onClose]);

  const handleClose = () => {
    console.log(LOG, 'close');
    setCameraReady(false);
    onClose();
  };

  if (!visible) {
    return null;
  }

  if (!permission) {
    return (
      <Modal visible animationType="slide" onRequestClose={handleClose}>
        <View style={styles.centered}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      </Modal>
    );
  }

  if (!permission.granted) {
    return (
      <Modal visible animationType="slide" onRequestClose={handleClose}>
        <View style={styles.centered}>
          <Text style={styles.permissionText}>
            Camera access is required for face verification.
          </Text>
          <Button label="Grant camera access" onPress={requestPermission} />
          <Button label="Cancel" variant="outline" onPress={handleClose} style={styles.cancelBtn} />
        </View>
      </Modal>
    );
  }

  return (
    <Modal visible animationType="slide" onRequestClose={handleClose}>
      <View style={styles.container}>
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          facing="front"
          mirror
          onCameraReady={() => {
            console.log(LOG, 'onCameraReady');
            setCameraReady(true);
          }}
        />

        <View style={styles.topBar}>
          <Pressable onPress={handleClose} style={styles.iconBtn} hitSlop={12}>
            <Ionicons name="close" size={28} color={colors.surface} />
          </Pressable>
          <Text style={styles.title}>Take a selfie</Text>
          <View style={styles.iconSpacer} />
        </View>

        <View style={styles.bottomBar}>
          <Button
            label={capturing ? 'Saving…' : 'Capture'}
            onPress={handleCapture}
            loading={capturing}
            disabled={!cameraReady || capturing}
            icon={<Ionicons name="camera" size={20} color={colors.secondary} />}
          />
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  topBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: spacing.xl + spacing.lg,
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.md,
    backgroundColor: 'rgba(0,0,0,0.35)',
  },
  title: {
    ...typography.label,
    color: colors.surface,
  },
  iconBtn: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconSpacer: {
    width: 40,
  },
  bottomBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: spacing.lg,
    paddingBottom: spacing.xl + spacing.lg,
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    padding: spacing.lg,
    gap: spacing.md,
    backgroundColor: colors.background,
  },
  permissionText: {
    ...typography.body,
    textAlign: 'center',
  },
  cancelBtn: {
    marginTop: spacing.sm,
  },
});
