export type WebhookItem = {
  id: string;
  url: string;
  events: string[];
  enabled: boolean;
  created_at?: string | null;
  updated_at?: string | null;
};

export type WebhookCreateResponse = WebhookItem & {
  secret: string;
};

export type WebhookListResponse = {
  items: WebhookItem[];
  total: number;
};

export type WebhookDeliveryItem = {
  id: string;
  webhook_id: string;
  event_type: string;
  status_code: number | null;
  response_time_ms: number | null;
  success: boolean;
  error: string | null;
  attempted_at: string | null;
  payload_preview: string | null;
};

export type WebhookDeliveryListResponse = {
  items: WebhookDeliveryItem[];
  total: number;
};
