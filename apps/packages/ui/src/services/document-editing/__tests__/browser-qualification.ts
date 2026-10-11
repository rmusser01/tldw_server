import { DocumentWorkerClient, admit, encode } from "../worker-client";
import { requestFor } from "./request-fixture";

export async function qualify(
  createWorker?: () => Worker,
  createFault: (name: string) => Worker = (name) =>
    new Worker(new URL("./fault-worker.js", import.meta.url), {
      type: "module",
      name,
    }),
) {
  const outcomes: Record<string, unknown> = {};
  const source = '{"n":9007199254740993,"x":-0}';
  const client = new DocumentWorkerClient(createWorker);
  const start = performance.now();
  const check = await client.check(requestFor(source));
  const format = await client.format(requestFor(source));
  if (
    !check.format_allowed ||
    format.state !== "formatted" ||
    !format.replacement?.includes("9007199254740993") ||
    !format.replacement?.includes("-0")
  )
    throw new Error("Positive worker failed");
  outcomes.positive = {
    check: check.state,
    format: format.state,
    elapsed: performance.now() - start,
  };
  for (const mode of [
    "stall",
    "crash",
    "malformed",
    "initstall",
    "cancel",
    "close",
  ]) {
    let terminated = 0;
    const owner = new DocumentWorkerClient(() => {
      const worker = createFault(
        ["cancel", "close"].includes(mode) ? "initstall" : mode,
      );
      const terminate = worker.terminate.bind(worker);
      worker.terminate = () => {
        terminated++;
        terminate();
      };
      return worker;
    });
    const began = performance.now();
    const pending = owner.check(requestFor('"E4_PRIVATE_SOURCE_MARKER"'));
    if (mode === "cancel" || mode === "close") {
      await new Promise((resolve) => setTimeout(resolve, 80));
      if (mode === "cancel") owner.cancel();
      else owner.close();
    }
    const result = await pending;
    const expected = ["stall", "initstall"].includes(mode)
      ? "resource_limited"
      : "failed";
    if (
      result.state !== expected ||
      result.complete ||
      terminated !== 1 ||
      JSON.stringify(result).includes("E4_PRIVATE_SOURCE_MARKER")
    )
      throw new Error(`Fault failed: ${mode}`);
    outcomes[mode] = {
      state: result.state,
      terminated,
      elapsed: performance.now() - began,
    };
    owner.close();
    if (!(await client.check(requestFor("{}"))).format_allowed)
      throw new Error("Replacement control failed");
  }
  const aliases = await client.check(
    requestFor(
      "a: &a [1]\nb: [" + Array(5000).fill("*a").join(",") + "]\n",
      "yaml",
    ),
  );
  const depth = await client.check(
    requestFor("[".repeat(101) + "0" + "]".repeat(101)),
  );
  const diagnostics = await client.check(requestFor("\n".repeat(101), "jsonl"));
  if (
    !aliases.format_allowed ||
    depth.state !== "resource_limited" ||
    diagnostics.diagnostics.length !== 100 ||
    !diagnostics.truncated ||
    diagnostics.complete
  )
    throw new Error("Resource controls failed");
  outcomes.resources = {
    aliases: aliases.state,
    depth: depth.state,
    diagnostics: diagnostics.diagnostics.length,
  };
  const boundary = requestFor('"' + "x".repeat(1024 * 1024 - 2) + '"');
  const admissionStarted = performance.now();
  if (admit(boundary)) throw new Error("Boundary refused");
  const admitted = performance.now();
  const frame = encode({ operation: "check", request: boundary });
  const encoded = performance.now();
  const boundaryResult = await client.check(boundary);
  if (!boundaryResult.format_allowed) throw new Error("Boundary worker failed");
  outcomes.boundary = {
    admissionMs: admitted - admissionStarted,
    encodeMs: encoded - admitted,
    frameBytes: frame.byteLength,
    totalMs: performance.now() - encoded,
  };
  for (let i = 0; i < 30; i++) {
    const lease = client.acquire(() => {
      throw new Error("Retired result escaped");
    });
    lease.submit(requestFor("{}", "json", { authority_id: `account-${i}` }));
    lease.retire();
    lease.submit(requestFor("{}"), true);
  }
  const finalLease = client.acquire(() => {});
  if ((await finalLease.format(requestFor("{}")))?.state !== "unchanged")
    throw new Error("Navigation control failed");
  outcomes.navigation = 30;
  client.close();
  let started = 0;
  const churn = new DocumentWorkerClient(() => {
    started++;
    return createFault("initstall");
  });
  const responsiveness: number[] = [];
  for (let i = 0; i < 6; i++) {
    const lease = churn.acquire(() => {
      throw new Error("Retired churn callback");
    });
    const request = requestFor("{}", "json", { authority_id: `churn-${i}` });
    lease.submit(request, true);
    const tick = performance.now();
    await new Promise((resolve) => setTimeout(resolve, 80));
    responsiveness.push(performance.now() - tick);
    if (started !== i + 1) throw new Error("Churn worker did not start");
    lease.retire();
    lease.submit(request, true); // Identical restored authority cannot revive a lease.
  }
  churn.close();
  if (Math.max(...responsiveness) > 500)
    throw new Error("Churn blocked the UI");
  outcomes.startupChurn = { started, responsivenessMs: responsiveness };
  return outcomes;
}
(globalThis as typeof globalThis & { e4Qualify: typeof qualify }).e4Qualify =
  qualify;
