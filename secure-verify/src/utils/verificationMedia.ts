import { Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const LOG = '[VerificationMedia]';

const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png'];

/**
 * Android: do NOT use launchCameraAsync — it starts a separate system Activity.
 * Expo Go / RN apps are killed under memory pressure; the JS promise never resolves
 * (Metro shows a full rebundle). Use in-app expo-camera instead (SelfieCameraModal).
 *
 * iOS: launchCameraAsync is stable enough to keep using the system camera UI.
 */
export const useInAppSelfieCamera = Platform.OS === 'android';

export type PickedImage = {
  uri: string;
  fileName: string | null;
};

export function getFileNameFromUri(uri: string): string | null {
  const segment = uri.split('/').pop();
  if (!segment) return null;
  return decodeURIComponent(segment.split('?')[0] ?? segment);
}

export function isAllowedImageType(fileName: string | null | undefined, mimeType?: string | null): boolean {
  if (mimeType) {
    const normalized = mimeType.toLowerCase();
    if (
      normalized === 'image/jpeg' ||
      normalized === 'image/jpg' ||
      normalized === 'image/png'
    ) {
      return true;
    }
  }

  if (!fileName) return true;

  const lower = fileName.toLowerCase();
  return ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function assetToPickedImage(
  asset: ImagePicker.ImagePickerAsset,
  defaultFileName: string
): PickedImage {
  return {
    uri: asset.uri,
    fileName: asset.fileName ?? getFileNameFromUri(asset.uri) ?? defaultFileName,
  };
}

/**
 * Recover a camera/gallery result if Android destroyed MainActivity but kept the process.
 * Does NOT work if LMKD killed the entire process (full Expo reload) — use in-app camera.
 */
export async function recoverPendingPickerResult(): Promise<{
  image: PickedImage | null;
  error: string | null;
}> {
  if (Platform.OS !== 'android') {
    return { image: null, error: null };
  }

  try {
    console.log(LOG, 'recoverPendingPickerResult: getPendingResultAsync');
    const pending = await ImagePicker.getPendingResultAsync();
    if (!pending) {
      console.log(LOG, 'recoverPendingPickerResult: null');
      return { image: null, error: null };
    }

    if ('code' in pending) {
      console.warn(LOG, 'recoverPendingPickerResult: error payload', pending);
      return { image: null, error: 'Previous photo capture was interrupted. Please try again.' };
    }

    console.log(LOG, 'recoverPendingPickerResult: got result', {
      canceled: pending.canceled,
      assetCount: pending.assets?.length ?? 0,
    });

    if (pending.canceled || !pending.assets?.[0]) {
      return { image: null, error: null };
    }

    const asset = pending.assets[0];
    return {
      image: assetToPickedImage(asset, 'selfie.jpg'),
      error: null,
    };
  } catch (err) {
    console.warn(LOG, 'recoverPendingPickerResult: exception', err);
    return { image: null, error: null };
  }
}

export async function pickDocumentImage(): Promise<{
  image: PickedImage | null;
  error: string | null;
}> {
  console.log(LOG, 'pickDocumentImage: start');
  const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (!permission.granted) {
    return {
      image: null,
      error: 'Photo library access is required to upload your document. Enable it in Settings.',
    };
  }

  try {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      quality: 0.9,
      aspect: [4, 3],
    });

    if (result.canceled || !result.assets[0]) {
      return { image: null, error: null };
    }

    const asset = result.assets[0];
    const fileName = asset.fileName ?? getFileNameFromUri(asset.uri);

    if (!isAllowedImageType(fileName, asset.mimeType)) {
      return {
        image: null,
        error: 'Please select a JPG, JPEG, or PNG image.',
      };
    }

    return { image: { uri: asset.uri, fileName }, error: null };
  } catch (err) {
    console.error(LOG, 'pickDocumentImage: exception', err);
    return {
      image: null,
      error: 'Could not select an image. Please try again.',
    };
  }
}

/** iOS only — Android must use SelfieCameraModal (expo-camera). */
export async function captureSelfieWithSystemCamera(): Promise<{
  image: PickedImage | null;
  error: string | null;
}> {
  console.log(LOG, 'captureSelfieWithSystemCamera: start');

  const permission = await ImagePicker.requestCameraPermissionsAsync();
  if (!permission.granted) {
    return {
      image: null,
      error: 'Camera access is required for face verification. Enable it in Settings.',
    };
  }

  try {
    console.log(LOG, 'captureSelfieWithSystemCamera: launchCameraAsync');
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      quality: 0.85,
      exif: false,
      cameraType: ImagePicker.CameraType.front,
    });

    console.log(LOG, 'captureSelfieWithSystemCamera: returned', {
      canceled: result.canceled,
      assetCount: result.assets?.length ?? 0,
    });

    if (result.canceled || !result.assets[0]) {
      return { image: null, error: null };
    }

    return {
      image: assetToPickedImage(result.assets[0], 'selfie.jpg'),
      error: null,
    };
  } catch (err) {
    console.error(LOG, 'captureSelfieWithSystemCamera: exception', err);
    return {
      image: null,
      error: 'Could not capture a photo. Please try again.',
    };
  }
}
