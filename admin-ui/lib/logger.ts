type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogContext {
  component?: string;
  requestId?: string;
  [key: string]: unknown;
}

const isProduction = process.env.NODE_ENV === 'production';

function formatLog(level: LogLevel, message: string, context?: LogContext): string {
  if (isProduction) {
    return JSON.stringify({
      timestamp: new Date().toISOString(),
      level,
      message,
      ...context,
    });
  }
  const prefix = `[${level.toUpperCase()}]`;
  const ctx = context ? ` ${JSON.stringify(context)}` : '';
  return `${prefix} ${message}${ctx}`;
}

function log(level: LogLevel, message: string, context?: LogContext): void {
  const formatted = formatLog(level, message, context);
  switch (level) {
    case 'error':
      console.error(formatted);
      break;
    case 'warn':
      console.warn(formatted);
      break;
    case 'debug':
      if (!isProduction) console.debug(formatted);
      break;
    default:
      console.log(formatted);
  }
}

export const logger = {
  debug: (message: string, context?: LogContext) => log('debug', message, context),
  info: (message: string, context?: LogContext) => log('info', message, context),
  warn: (message: string, context?: LogContext) => log('warn', message, context),
  error: (message: string, context?: LogContext) => log('error', message, context),
};
