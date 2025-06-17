# Multi-User Audio Transcription Architecture

## Executive Summary

While the fan-out design provides good horizontal scaling, a multi-user transcription system requires additional considerations for fairness, resource isolation, cost optimization, and user experience. This document proposes an improved architecture that better handles multiple concurrent users.

## Key Improvements Over Basic Fan-Out

### 1. **User-Aware Priority Queue System**

Instead of simple FIFO queues, implement a multi-level priority system:

```python
class UserAwarePriorityQueue:
    def __init__(self):
        self.user_queues = {}  # Per-user queues
        self.user_credits = {}  # Fair scheduling credits
        self.priority_levels = {
            'premium': 1.0,
            'standard': 0.5,
            'free': 0.1
        }
    
    async def enqueue(self, task: Task, user_id: str, priority: str = 'standard'):
        if user_id not in self.user_queues:
            self.user_queues[user_id] = asyncio.PriorityQueue()
            self.user_credits[user_id] = 1.0
        
        # Calculate dynamic priority based on user tier and recent usage
        effective_priority = self._calculate_priority(user_id, priority)
        await self.user_queues[user_id].put((effective_priority, task))
    
    def _calculate_priority(self, user_id: str, tier: str) -> float:
        base_priority = self.priority_levels[tier]
        usage_factor = self.user_credits[user_id]
        # Prevent starvation with aging
        age_bonus = time.time() - self.last_served.get(user_id, 0)
        return base_priority * usage_factor + (age_bonus / 3600)
```

### 2. **Resource Quota Management**

Implement per-user resource quotas to prevent abuse:

```python
class ResourceQuotaManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.quotas = {
            'free': {
                'daily_minutes': 30,
                'concurrent_jobs': 1,
                'max_file_size_mb': 25,
                'priority': 'low'
            },
            'standard': {
                'daily_minutes': 300,
                'concurrent_jobs': 3,
                'max_file_size_mb': 100,
                'priority': 'medium'
            },
            'premium': {
                'daily_minutes': 'unlimited',
                'concurrent_jobs': 10,
                'max_file_size_mb': 500,
                'priority': 'high'
            }
        }
    
    async def check_quota(self, user_id: str, audio_duration: int) -> QuotaCheck:
        user_tier = await self.get_user_tier(user_id)
        quota = self.quotas[user_tier]
        
        # Check concurrent jobs
        current_jobs = await self.redis.get(f"user:{user_id}:active_jobs")
        if current_jobs >= quota['concurrent_jobs']:
            return QuotaCheck(allowed=False, reason="Concurrent job limit reached")
        
        # Check daily usage
        if quota['daily_minutes'] != 'unlimited':
            used_today = await self.redis.get(f"user:{user_id}:minutes:{date.today()}")
            if used_today + audio_duration > quota['daily_minutes']:
                return QuotaCheck(allowed=False, reason="Daily quota exceeded")
        
        return QuotaCheck(allowed=True)
```

### 3. **Intelligent Model Routing**

Route users to appropriate models based on their needs and tier:

```python
class ModelRouter:
    def __init__(self):
        self.model_tiers = {
            'economy': {
                'models': ['tiny', 'base'],
                'gpu_fraction': 0.2,
                'batch_size': 8
            },
            'balanced': {
                'models': ['small', 'medium'],
                'gpu_fraction': 0.5,
                'batch_size': 4
            },
            'quality': {
                'models': ['large', 'large-v2'],
                'gpu_fraction': 1.0,
                'batch_size': 1
            }
        }
    
    async def route_request(self, user_request: TranscriptionRequest) -> ModelAllocation:
        # Consider user tier, file size, and language
        if user_request.user_tier == 'free':
            tier = 'economy'
        elif user_request.require_high_accuracy:
            tier = 'quality'
        else:
            tier = 'balanced'
        
        # Find available GPU with enough resources
        gpu = await self.find_available_gpu(self.model_tiers[tier]['gpu_fraction'])
        
        return ModelAllocation(
            model_name=self._select_model(tier, user_request.language),
            gpu_id=gpu.id,
            batch_size=self.model_tiers[tier]['batch_size']
        )
```

### 4. **Batching for Efficiency**

Implement intelligent batching to maximize GPU utilization:

```python
class BatchingTranscriptionWorker:
    def __init__(self, model_name: str, gpu_id: int, batch_size: int = 4):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.batch_timeout = 5.0  # seconds
        self.pending_batch = []
        
    async def process_requests(self):
        while True:
            # Collect requests until batch is full or timeout
            batch = await self._collect_batch()
            
            if batch:
                # Process multiple files simultaneously on GPU
                results = await self._batch_transcribe(batch)
                
                # Distribute results back to users
                for request, result in zip(batch, results):
                    await self._send_result(request.user_id, result)
    
    async def _collect_batch(self) -> List[TranscriptionRequest]:
        batch = []
        deadline = time.time() + self.batch_timeout
        
        while len(batch) < self.batch_size and time.time() < deadline:
            try:
                timeout = deadline - time.time()
                request = await asyncio.wait_for(
                    self.queue.get(), 
                    timeout=max(0.1, timeout)
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break
        
        return batch
```

### 5. **Edge Processing Option**

Offer client-side processing for privacy-conscious users:

```python
class HybridProcessingRouter:
    def __init__(self):
        self.edge_capable_clients = set()
        
    async def route_processing(self, request: ProcessingRequest) -> ProcessingPlan:
        if request.privacy_mode and request.client_id in self.edge_capable_clients:
            # Return edge processing instructions
            return ProcessingPlan(
                mode='edge',
                model_url=self._get_edge_model_url(request.language),
                processing_script=self._get_wasm_processor(),
                fallback_to_cloud=request.allow_fallback
            )
        else:
            # Use cloud processing
            return ProcessingPlan(
                mode='cloud',
                queue='transcription_queue',
                priority=self._calculate_priority(request)
            )
```

### 6. **Streaming Results with WebSocket**

Implement real-time streaming for better user experience:

```python
class StreamingTranscriptionService:
    def __init__(self):
        self.active_streams = {}
        
    async def handle_websocket(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        stream_id = str(uuid.uuid4())
        
        try:
            # Receive audio chunks
            async for audio_chunk in websocket.iter_bytes():
                await self.process_chunk(stream_id, audio_chunk)
                
                # Send partial results
                if partial_result := await self.get_partial_result(stream_id):
                    await websocket.send_json({
                        'type': 'partial',
                        'text': partial_result.text,
                        'timestamp': partial_result.timestamp
                    })
            
            # Send final result
            final_result = await self.finalize_transcription(stream_id)
            await websocket.send_json({
                'type': 'final',
                'text': final_result.text,
                'segments': final_result.segments
            })
            
        finally:
            await self.cleanup_stream(stream_id)
```

### 7. **Cost-Optimized Architecture**

Implement smart routing between different processing backends:

```python
class CostOptimizedProcessor:
    def __init__(self):
        self.backends = {
            'on_demand_gpu': {
                'cost_per_minute': 0.10,
                'startup_time': 60,
                'min_duration': 300  # 5 minutes minimum
            },
            'reserved_gpu': {
                'cost_per_minute': 0.03,
                'startup_time': 0,
                'availability': self.check_reserved_availability
            },
            'cpu_cluster': {
                'cost_per_minute': 0.01,
                'startup_time': 0,
                'speed_multiplier': 0.1  # 10x slower
            },
            'spot_gpu': {
                'cost_per_minute': 0.02,
                'startup_time': 120,
                'interruption_risk': 0.1
            }
        }
    
    async def select_backend(self, job: TranscriptionJob) -> Backend:
        # Calculate cost-effectiveness for each backend
        scores = {}
        
        for backend_name, backend in self.backends.items():
            if backend_name == 'cpu_cluster' and job.duration > 600:
                continue  # Skip CPU for long files
            
            cost = self._calculate_total_cost(backend, job)
            time = self._calculate_total_time(backend, job)
            
            # Factor in user preferences
            if job.user_preferences.optimize_for == 'cost':
                score = 1 / cost
            elif job.user_preferences.optimize_for == 'speed':
                score = 1 / time
            else:  # balanced
                score = 1 / (cost * time)
            
            scores[backend_name] = score
        
        return max(scores, key=scores.get)
```

### 8. **Improved Architecture Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                     │
└─────────────────┬───────────────────┬───────────────────────┘
                  │                   │
          ┌───────▼──────┐    ┌───────▼──────┐
          │   API Gateway │    │  WebSocket   │
          │   (FastAPI)   │    │   Gateway    │
          └───────┬──────┘    └───────┬──────┘
                  │                   │
         ┌────────▼───────────────────▼────────┐
         │        Routing & Auth Layer         │
         │  - User Authentication              │
         │  - Quota Management                 │
         │  - Request Routing                  │
         └────────┬───────────────────┬────────┘
                  │                   │
    ┌─────────────▼──────┐  ┌─────────▼────────────┐
    │   Priority Queue   │  │  Edge Processing     │
    │   Manager          │  │  Coordinator         │
    │ - User Fairness    │  │ - WASM Delivery      │
    │ - SLA Guarantees   │  │ - Client Orchestr.   │
    └─────────────┬──────┘  └──────────────────────┘
                  │
    ┌─────────────▼───────────────────────────┐
    │         Resource Scheduler              │
    │  - GPU Allocation                       │
    │  - Model Routing                        │
    │  - Batch Optimization                    │
    └──────┬──────────┬─────────┬────────────┘
           │          │         │
    ┌──────▼───┐ ┌────▼───┐ ┌──▼─────┐
    │ Reserved │ │  Spot  │ │  CPU   │
    │   GPUs   │ │  GPUs  │ │ Cluster│
    │ (Premium)│ │ (Std)  │ │ (Free) │
    └──────────┘ └────────┘ └────────┘
```

## Implementation Recommendations

### 1. **Start with User Isolation**
- Implement per-user queues and quotas first
- This provides immediate fairness benefits

### 2. **Add Intelligent Routing**
- Route based on user tier and job characteristics
- Implement gradually with A/B testing

### 3. **Implement Batching**
- Start with simple time-based batching
- Evolve to dynamic batching based on GPU utilization

### 4. **Cost Optimization**
- Begin with reserved vs on-demand routing
- Add spot instances once system is stable

### 5. **Edge Processing**
- Start as beta feature for premium users
- Use ONNX or TensorFlow.js for browser execution

## Key Metrics to Track

1. **User Experience**
   - Time to first result (TTFR)
   - Queue wait time by user tier
   - Transcription accuracy by model

2. **System Efficiency**
   - GPU utilization percentage
   - Cost per transcription minute
   - Queue depth by priority level

3. **Business Metrics**
   - User tier distribution
   - Quota utilization rates
   - Revenue per GPU hour

## Conclusion

This architecture provides:
- **Better multi-user fairness** through priority queuing
- **Cost optimization** via intelligent routing
- **Improved user experience** with streaming and edge options
- **Resource efficiency** through batching and quotas
- **Scalability** with tier-based resource allocation

The key insight is that multi-user transcription isn't just about scaling horizontally—it's about intelligently managing shared resources while providing differentiated service levels.