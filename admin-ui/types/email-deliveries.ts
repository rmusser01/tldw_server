export type EmailDeliveryStatus = 'sent' | 'failed' | 'skipped';

export type EmailDeliveryItem = {
  id: string;
  recipient: string;
  subject: string;
  template: string | null;
  status: EmailDeliveryStatus;
  error: string | null;
  sent_at: string;
};

export type EmailDeliveryListResponse = {
  items: EmailDeliveryItem[];
  total: number;
  limit: number;
  offset: number;
};
