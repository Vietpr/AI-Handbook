export const PRIORITY_VI: Record<string, string> = {
  Essential: 'Cốt lõi',
  High: 'Cao',
  Medium: 'Vừa',
  Low: 'Thấp',
  Optional: 'Tùy chọn',
};

export const TYPE_VI: Record<string, string> = {
  Concept: 'Khái niệm',
  Algorithm: 'Thuật toán',
  LeetCode: 'LeetCode',
};

export const domainKey = (domain: string) => domain.toLowerCase().replaceAll(' ', '-');
