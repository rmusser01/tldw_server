import type { CheckResult, DocumentRequest, FormatResult } from "./types";

interface WorkerLane {
  check(request: DocumentRequest): Promise<CheckResult>;
  format(request: DocumentRequest): Promise<FormatResult>;
  cancel(): void;
}
interface Pending {
  request: DocumentRequest;
  operation: "check" | "format";
  generation: number;
  ready: number;
  settle?: (result: FormatResult | null) => void;
}

/** One active-editor lease; revocation permanently retires pending authority. */
export class DocumentCheckScheduler {
  private pending: Pending | null = null;
  private active: Pending | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private generation = 0;
  private retired = false;

  constructor(
    private worker: WorkerLane,
    private onResult: ((result: CheckResult) => void) | null,
  ) {}

  submit(request: DocumentRequest, immediate = false): void {
    this.enqueue(request, "check", immediate ? 0 : 400);
  }

  format(request: DocumentRequest): Promise<FormatResult | null> {
    if (this.retired) return Promise.resolve(null);
    return new Promise((resolve) =>
      this.enqueue(request, "format", 0, resolve),
    );
  }

  private enqueue(
    request: DocumentRequest,
    operation: Pending["operation"],
    delay: number,
    settle?: Pending["settle"],
  ): void {
    if (this.retired) return;
    this.pending?.settle?.(null);
    this.pending = {
      request,
      operation,
      generation: ++this.generation,
      ready: performance.now() + delay,
      settle,
    };
    this.schedule();
  }

  private schedule(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    if (this.retired || this.active || !this.pending) return;
    this.timer = setTimeout(
      () => {
        this.timer = null;
        void this.run();
      },
      Math.max(0, this.pending.ready - performance.now()),
    );
  }

  private async run(): Promise<void> {
    const item = this.pending;
    if (!item || this.retired || this.active) return;
    this.pending = null;
    this.active = item;
    try {
      const result = await this.worker[item.operation](item.request);
      if (!this.retired && item.generation === this.generation) {
        if (item.operation === "check") this.onResult?.(result as CheckResult);
        else item.settle?.(result as FormatResult);
      }
    } finally {
      item.settle?.(null);
      this.active = null;
      this.schedule();
    }
  }

  retire(): void {
    if (this.retired) return;
    this.retired = true;
    this.generation++;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    this.pending?.settle?.(null);
    this.pending = null;
    this.active?.settle?.(null);
    this.onResult = null;
    this.worker.cancel();
  }

  close(): void {
    this.retire();
  }
}
