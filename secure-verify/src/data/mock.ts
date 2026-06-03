export type VerificationStatus = 'approved' | 'pending' | 'rejected';

export interface VerificationRecord {
  id: string;
  title: string;
  date: string;
  status: VerificationStatus;
  documentType: string;
}

export interface UserProfile {
  name: string;
  email: string;
  avatarInitials: string;
  memberSince: string;
}

export const mockUser: UserProfile = {
  name: 'Alex Morgan',
  email: 'alex.morgan@email.com',
  avatarInitials: 'AM',
  memberSince: 'Jan 2025',
};

export const mockStats = {
  totalVerifications: 12,
  approved: 9,
  pending: 2,
  successRate: 94,
};

export const mockRecentVerifications: VerificationRecord[] = [
  {
    id: 'SV-2025-0847',
    title: 'Passport Verification',
    date: 'May 28, 2025',
    status: 'approved',
    documentType: 'Passport',
  },
  {
    id: 'SV-2025-0831',
    title: "Driver's License",
    date: 'May 22, 2025',
    status: 'approved',
    documentType: 'License',
  },
  {
    id: 'SV-2025-0819',
    title: 'National ID Card',
    date: 'May 15, 2025',
    status: 'pending',
    documentType: 'National ID',
  },
];

export const mockHistory: VerificationRecord[] = [
  ...mockRecentVerifications,
  {
    id: 'SV-2025-0792',
    title: 'Passport Verification',
    date: 'May 8, 2025',
    status: 'approved',
    documentType: 'Passport',
  },
  {
    id: 'SV-2025-0764',
    title: 'Residence Permit',
    date: 'Apr 30, 2025',
    status: 'rejected',
    documentType: 'Permit',
  },
  {
    id: 'SV-2025-0741',
    title: "Driver's License",
    date: 'Apr 18, 2025',
    status: 'approved',
    documentType: 'License',
  },
  {
    id: 'SV-2025-0718',
    title: 'National ID Card',
    date: 'Apr 5, 2025',
    status: 'approved',
    documentType: 'National ID',
  },
];

export const settingsSections = [
  {
    title: 'Account',
    items: [
      { id: 'profile', label: 'Profile', icon: 'person-outline' as const, value: 'Alex Morgan' },
      {
        id: 'notifications',
        label: 'Notifications',
        icon: 'notifications-outline' as const,
        value: 'On',
      },
    ],
  },
  {
    title: 'Security',
    items: [
      { id: 'security', label: 'Security', icon: 'shield-checkmark-outline' as const, value: 'Strong' },
      { id: 'theme', label: 'Theme', icon: 'color-palette-outline' as const, value: 'Light' },
    ],
  },
  {
    title: 'About',
    items: [
      { id: 'about', label: 'About Secure Verify', icon: 'information-circle-outline' as const, value: 'v1.0.0' },
      { id: 'privacy', label: 'Privacy Policy', icon: 'document-text-outline' as const },
      { id: 'terms', label: 'Terms of Service', icon: 'reader-outline' as const },
    ],
  },
];

export function generateVerificationId(): string {
  const num = Math.floor(1000 + Math.random() * 9000);
  return `SV-2025-${num}`;
}