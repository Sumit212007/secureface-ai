export const APP_NAME = 'Secure Verify';

export const VERIFICATION_STEPS = [
  { id: 1, title: 'Document Upload', subtitle: 'Upload your government ID' },
  {
    id: 2,
    title: 'Face Verification & Liveness',
    subtitle: 'Take a selfie and complete liveness checks',
  },
] as const;

export const HISTORY_FILTERS = ['All', 'Approved', 'Pending', 'Rejected'] as const;
