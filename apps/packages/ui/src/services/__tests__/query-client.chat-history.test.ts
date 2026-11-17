import { describe, expect, it, vi } from 'vitest';
import { createQueryClient, invalidateServerChatHistory } from '../query-client';

it('retires a pending first finder page before invalidation so it cannot become fresh', async () => {
  const client = createQueryClient();
  const key = ['serverChatHistory', { ownerKey: 'original' }];
  let release!: (rows: string[]) => void;
  const read = client
    .fetchQuery({
      queryKey: key,
      queryFn: () =>
        new Promise<string[]>((resolve) => {
          release = resolve;
        }),
    })
    .catch(() => undefined);
  client.setQueryData(['mediaHistory'], ['unchanged']);
  await invalidateServerChatHistory(client);
  release(['obsolete empty overview']);
  await read;
  expect(client.getQueryData(key)).toBeUndefined();
  expect(client.getQueryState(key)?.isInvalidated).toBe(true);
  expect(client.getQueryData(['mediaHistory'])).toEqual(['unchanged']);
  client.clear();
});

describe('original account guard during finder cancellation', () => {
  it('starts no cancellation or reads for an already retired owner', async () => {
    const client = createQueryClient();
    const cancel = vi.spyOn(client, 'cancelQueries');
    const refresh = vi.spyOn(client, 'invalidateQueries');
    await invalidateServerChatHistory(client, () => false);
    expect(cancel).not.toHaveBeenCalled();
    expect(refresh).not.toHaveBeenCalled();
    client.clear();
  });

  it('does not start replacement reads when the original account retires during cancellation', async () => {
    const client = createQueryClient();
    const originalAccount = new AbortController();
    let release!: () => void;
    vi.spyOn(client, 'cancelQueries').mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          release = resolve;
        })
    );
    const refresh = vi.spyOn(client, 'invalidateQueries');
    const pending = invalidateServerChatHistory(client, () => !originalAccount.signal.aborted);
    originalAccount.abort();
    release();
    await pending;
    expect(refresh).not.toHaveBeenCalled();
    client.clear();
  });
});
