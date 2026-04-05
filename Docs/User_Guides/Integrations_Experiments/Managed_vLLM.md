# Managed vLLM

Managed `vLLM` lets `tldw_server` own the lifecycle of one or more external `vLLM` servers without embedding those runtimes inside the API process. The control plane stores durable instance records, starts and stops runtimes locally or over SSH, and routes chat or embeddings requests to a selected instance at request time.

## What Shipped

- Persistent multi-instance registry backed by `Databases/vllm_instances.db` by default.
- Local and SSH execution modes.
- Admin API at `/api/v1/llm/providers/vllm/instances*`.
- Admin page at `/admin/vllm`.
- Request-scoped routing with `provider_instance_id`.
- Jobs-backed `start`, `stop`, `restart`, and `probe`.
- Provider listing enrichment through `GET /api/v1/llm/providers`.

## Important Constraints

- `tldw_server` must be able to reach every managed instance at its resolved `base_url`. This feature does not proxy inference traffic.
- Supported lifecycle executors today are `local` and `ssh`. `agent` is reserved but not implemented yet.
- SSH mode requires a remote launcher binary or script. The server does not background raw shell commands over SSH.
- `auth.secret_ref` in SSH transport config is resolved as an environment variable name on the `tldw_server` host. The environment variable value must be the path to the SSH private key file used with `ssh -i`.

## Instance Model

Each instance record stores:

- identity: `instance_id`, `name`
- execution: `execution_mode`, `transport_config`
- launch: `launch_spec`
- routing: `routing_policy`
- capabilities: `declared_capabilities`, `probed_capabilities`, `effective_capabilities`
- runtime state: `desired_state`, `observed_state`, `last_known_base_url`, `last_error`, `executor_handle`

The launch and transport specs are durable. Runtime fields are reconciled from lifecycle jobs plus startup and periodic probing.

## Admin API

### CRUD and default route

- `POST /api/v1/llm/providers/vllm/instances`
- `GET /api/v1/llm/providers/vllm/instances`
- `GET /api/v1/llm/providers/vllm/instances/{instance_id}`
- `PATCH /api/v1/llm/providers/vllm/instances/{instance_id}`
- `DELETE /api/v1/llm/providers/vllm/instances/{instance_id}`
- `POST /api/v1/llm/providers/vllm/default`

### Lifecycle actions

- `POST /api/v1/llm/providers/vllm/instances/{instance_id}/start`
- `POST /api/v1/llm/providers/vllm/instances/{instance_id}/stop`
- `POST /api/v1/llm/providers/vllm/instances/{instance_id}/restart`
- `POST /api/v1/llm/providers/vllm/instances/{instance_id}/probe`

Lifecycle endpoints return `202` with job metadata. They enqueue work instead of blocking until the model is ready.

## Example Create Payloads

### Local instance

```json
{
  "name": "local-qwen-vl",
  "execution_mode": "local",
  "transport_config": {
    "workdir": ".",
    "log_dir": "Databases/vllm_logs/local-qwen-vl"
  },
  "launch_spec": {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "served_model_name": "qwen2.5-vl",
    "host": "127.0.0.1",
    "port": 8002,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9
  },
  "routing_policy": {
    "is_default": true
  },
  "declared_capabilities": {
    "chat": true,
    "vision": true
  }
}
```

### SSH instance

```json
{
  "name": "embed-gpu-a",
  "execution_mode": "ssh",
  "transport_config": {
    "host": "gpu-a.internal",
    "port": 22,
    "user": "mlops",
    "launcher_path": "/usr/local/bin/tldw-vllm-launcher",
    "auth": {
      "secret_ref": "VLLM_GPU_A_SSH_KEY_PATH",
      "strict_host_key_checking": true
    },
    "base_url": "http://gpu-a.internal:8010/v1"
  },
  "launch_spec": {
    "model": "BAAI/bge-m3",
    "served_model_name": "bge-m3",
    "port": 8010
  },
  "declared_capabilities": {
    "embeddings": true
  }
}
```

`transport_config.user` is the canonical field. `transport_config.username` is also accepted for compatibility with older payloads.

## SSH Launcher Contract

The SSH executor builds explicit launcher commands like:

```text
/usr/local/bin/tldw-vllm-launcher start --instance-id <id> --json-spec <json>
```

and

```text
/usr/local/bin/tldw-vllm-launcher stop --instance-id <id> --remote-pid <pid>
```

The remote launcher is expected to start or stop the `vllm serve ...` process and return JSON metadata, such as a remote pid, on stdout. `tldw_server` does not rely on `nohup`, `&`, or shell backgrounding to manage remote runtimes.

## Routing Requests

### Chat

Target managed `vLLM` explicitly by setting the provider and instance id:

```json
{
  "api_provider": "vllm",
  "provider_instance_id": "local-qwen-vl",
  "model": "ignored-when-managed-route-overrides",
  "messages": [
    {"role": "user", "content": "Describe the image."}
  ]
}
```

If no `provider_instance_id` is supplied, the managed default `vLLM` instance is used when one is configured. If neither a managed default nor a managed instance id is present, the legacy single-endpoint `vllm_api` config still applies.

### Embeddings

```json
{
  "provider": "vllm",
  "provider_instance_id": "embed-gpu-a",
  "model": "ignored-when-managed-route-overrides",
  "input": "The food was delicious and the waiter was friendly."
}
```

For embeddings, `provider_instance_id` can also imply managed `vLLM` routing even when `provider` is omitted, but using `provider: "vllm"` is the clearer contract.

## Capabilities and Health

Managed `vLLM` uses three capability layers:

- `declared_capabilities`: operator intent
- `probed_capabilities`: what probes observed
- `effective_capabilities`: what routing is allowed to use

Current behavior:

- `chat` can route when it is declared and any present probe remains positive.
- `embeddings`, `vision`, `audio`, and `multimodal` only become effective when they are both declared and positively probed.
- Requests fail fast when the selected instance is missing the required effective capability.
- Request routing only uses managed instances whose `observed_state` is `healthy`. Instances in `starting`, `stopped`, `stopping`, `failed`, or `unhealthy` are rejected before inference dispatch.

Health probes also update `last_known_base_url`, `last_error`, and `observed_state`.

Startup behavior:

- A freshly started instance stays in `starting` if the first probe misses during cold boot.
- `VLLM_MANAGEMENT_STARTUP_TIMEOUT_SECONDS` bounds how long `starting` can persist before later probes mark the instance `unhealthy`.
- The managed vLLM reconciler loop probes persisted records on startup and then continues periodically, so slow boots can converge to `healthy` without a manual probe.

Provider listing behavior:

- Managed provider metadata only advertises a managed default when `default_instance_id` is explicitly configured.
- If no managed default is configured, top-level `/llm/providers` data falls back to the legacy `vllm_model` / `vllm_api_IP` config when present.
- If neither a managed default nor a legacy fallback is configured, `/llm/providers` leaves the vLLM default unset instead of inventing one from the first stored managed instance.

## Worker and Reconciler Flags

- `VLLM_INSTANCES_DB_PATH`: override the registry database path.
- `VLLM_MANAGEMENT_WORKER_ENABLED`: enable the managed `vLLM` Jobs worker.
- `VLLM_MANAGEMENT_STARTUP_RECONCILE_ENABLED`: enable the managed `vLLM` reconciler loop that probes stored instances on startup and then periodically afterward.
- `VLLM_MANAGEMENT_STARTUP_TIMEOUT_SECONDS`: maximum time an instance may remain in `starting` before later probes mark it `unhealthy`.
- `VLLM_MANAGEMENT_JOBS_QUEUE`: override the queue name used for lifecycle jobs.
- `VLLM_MANAGEMENT_WORKER_ID`: override the worker id string.

Without the Jobs worker, lifecycle API calls still enqueue work but nothing will execute those jobs.
