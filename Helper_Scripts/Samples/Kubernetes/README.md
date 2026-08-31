# Kubernetes Samples

These manifests provide a simple, non-HA deployment of tldw_server with Postgres and Redis.
They are meant as starting points - review and adapt for your cluster (Ingress class, TLS secrets, images, resources).

## Files
- `namespace.yaml` - Namespace `tldw`
- `postgres-statefulset.yaml` - Headless Service + StatefulSet for PostgreSQL
- `redis-deployment.yaml` - Service + Deployment for Redis
- `app-configmap.yaml` - Non-sensitive app configuration (mode, CORS, logging)
- `app-secret.yaml` - Sensitive values (JWT/DB URL/password)
- `tldw-app-deployment.yaml` - App Deployment + Service (8000)
- `ingress.yaml` - NGINX Ingress with long timeouts + TLS

## Prerequisites
- NGINX Ingress Controller (or adjust for your ingress class)
- A TLS secret for your domain (or use cert-manager and change annotations)
- An image for the app (`tldw-server:prod` is a placeholder - replace with your registry)

## Quick Start (apply order)
```bash
# 1) Namespace
kubectl apply -f Samples/Kubernetes/namespace.yaml

# 2) Data services
kubectl apply -f Samples/Kubernetes/postgres-statefulset.yaml
kubectl apply -f Samples/Kubernetes/redis-deployment.yaml

# 3) App config + secrets (EDIT FIRST)
#   - Set JWT_SECRET_KEY (multi_user) or SINGLE_USER_API_KEY (single_user)
#   - Set DATABASE_URL to point at your Postgres service
kubectl apply -f Samples/Kubernetes/app-configmap.yaml
kubectl apply -f Samples/Kubernetes/app-secret.yaml

# 4) App + Service
#    Edit tldw-app-deployment.yaml to use your image (e.g., ghcr.io/your-org/tldw-server:prod)
kubectl apply -f Samples/Kubernetes/tldw-app-deployment.yaml

# 5) Ingress (set your.domain.com and TLS secret name)
kubectl apply -f Samples/Kubernetes/ingress.yaml
```

## Configuration Notes
- Auth mode: `app-configmap.yaml` defaults to `multi_user`. Set secrets/env accordingly.
- CORS: Set `ALLOWED_ORIGINS` to your HTTPS origin (e.g., `https://your.domain.com`).
- Postgres: The sample uses `POSTGRES_DB=tldw_users`, `POSTGRES_USER=tldw_user`, password from `app-secret.yaml`.
- Liveness is the minimal public `/health` contract. Readiness executes inside the
  application container against loopback-only `/internal/ready`; detailed readiness
  is not remotely anonymous.

## Verify
```bash
kubectl -n tldw get pods,svc,ingress
kubectl -n tldw logs deploy/tldw-app -f
curl -k https://your.domain.com/health
```

## Cleanup
```bash
kubectl delete ns tldw
```

## Next Steps
- Harden Postgres (storage class, backups, auth)
- Move secrets to an external secret manager
- Add HorizontalPodAutoscaler for the app
- Configure cert-manager to issue TLS certificates automatically
