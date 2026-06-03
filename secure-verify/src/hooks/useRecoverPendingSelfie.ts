import { useEffect, useRef } from 'react';
import { AppState, type AppStateStatus } from 'react-native';

import { PickedImage, recoverPendingPickerResult } from '@/utils/verificationMedia';

const LOG = '[useRecoverPendingSelfie]';

/**
 * After launchCameraAsync on Android, the OS may destroy MainActivity while the
 * system camera is open. If the process survives, expo-image-picker stores the
 * result for getPendingResultAsync — call this on mount and every AppState "active".
 */
export function useRecoverPendingSelfie(
  enabled: boolean,
  onRecovered: (image: PickedImage) => void
) {
  const onRecoveredRef = useRef(onRecovered);
  onRecoveredRef.current = onRecovered;

  const inFlightRef = useRef(false);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const tryRecover = async (reason: string) => {
      if (inFlightRef.current) {
        return;
      }
      inFlightRef.current = true;
      console.log(LOG, 'tryRecover', reason);

      try {
        const { image, error } = await recoverPendingPickerResult();
        if (error) {
          console.warn(LOG, 'recovery error', error);
        }
        if (image) {
          console.log(LOG, 'recovered selfie');
          onRecoveredRef.current(image);
        }
      } finally {
        inFlightRef.current = false;
      }
    };

    void tryRecover('mount');

    const subscription = AppState.addEventListener('change', (nextState: AppStateStatus) => {
      if (nextState === 'active') {
        void tryRecover('app-active');
      }
    });

    return () => subscription.remove();
  }, [enabled]);
}
