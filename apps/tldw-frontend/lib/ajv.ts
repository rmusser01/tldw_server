import type { JsonSchema } from './schema';

interface AjvError {
  instancePath?: string;
  schemaPath?: string;
  message?: string;
}

interface AjvModule {
  default?: new (options: Record<string, unknown>) => AjvInstance;
}

interface AjvInstance {
  compile: (schema: JsonSchema) => AjvValidateFunction;
}

interface AjvValidateFunction {
  (data: unknown): boolean;
  errors?: AjvError[] | null;
}

export async function validateWithAjv(data: unknown, schema?: JsonSchema, options: { strict?: boolean } = {}): Promise<string[]> {
  if (!schema) return [];
  if (options.strict && (schema as Record<string, unknown>).$async) throw new Error('Asynchronous workflow input schemas are not supported');
  let ajv: AjvInstance;
  try {
    const mod = await import('ajv') as AjvModule;
    const Ajv = mod.default;
    if (!Ajv) {
      throw new Error('missing default export');
    }
    ajv = new Ajv({ allErrors: true, allowUnionTypes: true, strict: options.strict ?? false });
  } catch (err) {
    // ajv not installed or failed to initialize; surface to caller
    if (typeof console !== 'undefined') {
      console.warn('[validateWithAjv] AJV initialization failed – schema validation cannot proceed');
    }
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`AJV unavailable: ${message}`);
  }
  const validate = ajv.compile(schema);
  const ok = validate(data);
  if (ok) return [];
  const errs = (validate.errors || []).map((e: AjvError) => {
    const path = e.instancePath || e.schemaPath;
    const msg = e.message ?? 'invalid';
    return path ? `${path}: ${msg}` : msg;
  });
  return errs;
}
