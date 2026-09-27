export const VN_COMMAND_STORAGE_KEY = 'tldw:vn-generation:pending:v1';

export interface VNCommandScope {
  server: string;
  principal: string;
}

export interface VNPendingCommand {
  packId: number;
  slotId?: number;
  request: { idempotency_key: string; source_batch_id?: number };
}

export class VNRecoveryStorageError extends Error {
  constructor(public readonly unreadable = false) {
    super(unreadable
      ? 'Saved generation requests could not be read. Check server generation status before discarding them.'
      : 'Recovery storage is unavailable. No new generation request was sent. Enable session storage and try again.');
  }
}

const isId = (value: unknown): value is number => Number.isSafeInteger(value) && (value as number) > 0;
const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
const onlyKeys = (value: Record<string, unknown>, keys: string[]) => Object.keys(value).every((key) => keys.includes(key));

export function createVNCommandScope(server: string, principal: unknown): VNCommandScope {
  const url = new URL(server, window.location.origin);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash ||
      !isId(principal)) throw new Error('Current server and account could not be verified.');
  return { server: `${url.origin}${url.pathname.replace(/\/+$/, '')}`, principal: String(principal) };
}

export function sameVNCommandScope(left: VNCommandScope | null, right: VNCommandScope): boolean {
  return left?.server === right.server && left.principal === right.principal;
}

function validCommand(value: unknown): value is VNPendingCommand {
  if (!object(value) || !onlyKeys(value, ['packId', 'slotId', 'request']) || !isId(value.packId) ||
      (value.slotId !== undefined && !isId(value.slotId)) || !object(value.request) ||
      !onlyKeys(value.request, ['idempotency_key', 'source_batch_id'])) return false;
  const request = value.request;
  return typeof request.idempotency_key === 'string' && /^[A-Za-z0-9._~-]{16,200}$/.test(request.idempotency_key) &&
    (request.source_batch_id === undefined || (isId(value.slotId) && isId(request.source_batch_id)));
}

export function readVNCommands(scope: VNCommandScope): VNPendingCommand[] {
  try {
    const raw = window.sessionStorage.getItem(VN_COMMAND_STORAGE_KEY);
    if (raw === null) return [];
    if (raw.length > 32768) throw new VNRecoveryStorageError(true);
    const value: unknown = JSON.parse(raw);
    if (!object(value) || !onlyKeys(value, ['version', 'scope', 'commands']) || value.version !== 1 ||
        !object(value.scope) || !onlyKeys(value.scope, ['server', 'principal']) ||
        typeof value.scope.server !== 'string' || typeof value.scope.principal !== 'string' ||
        !Array.isArray(value.commands) || value.commands.length > 64 || !value.commands.every(validCommand) ||
        new Set(value.commands.map((command) => command.packId)).size !== value.commands.length) {
      throw new VNRecoveryStorageError(true);
    }
    let savedScope: VNCommandScope;
    try {
      savedScope = createVNCommandScope(value.scope.server, Number(value.scope.principal));
    } catch {
      throw new VNRecoveryStorageError(true);
    }
    if (value.scope.server !== savedScope.server || value.scope.principal !== savedScope.principal) {
      throw new VNRecoveryStorageError(true);
    }
    if (!sameVNCommandScope(savedScope, scope)) {
      window.sessionStorage.removeItem(VN_COMMAND_STORAGE_KEY);
      return [];
    }
    return value.commands;
  } catch (error) {
    if (error instanceof VNRecoveryStorageError) throw error;
    if (error instanceof SyntaxError) throw new VNRecoveryStorageError(true);
    throw new VNRecoveryStorageError();
  }
}

export function writeVNCommands(scope: VNCommandScope, commands: VNPendingCommand[]): void {
  if (commands.length > 64 || !commands.every(validCommand) || new Set(commands.map((command) => command.packId)).size !== commands.length) {
    throw new VNRecoveryStorageError(true);
  }
  try {
    if (!commands.length) window.sessionStorage.removeItem(VN_COMMAND_STORAGE_KEY);
    else window.sessionStorage.setItem(VN_COMMAND_STORAGE_KEY, JSON.stringify({ version: 1, scope, commands }));
  } catch {
    throw new VNRecoveryStorageError();
  }
}

export function clearVNCommands(): void {
  try {
    window.sessionStorage.removeItem(VN_COMMAND_STORAGE_KEY);
  } catch {
    throw new VNRecoveryStorageError();
  }
}
