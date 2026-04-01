# PersonaPlex Hosting Cost Analysis

## Service Pricing Model

| Parameter            | Value              |
| -------------------- | ------------------ |
| Revenue per user     | $20/month          |
| Target margin        | 80-85%             |
| **Max cost per user**| **$3-4/month**     |
| Usage per user       | 2 hr/day = 60 GPU-hrs/month |

## Model Requirements

- **Model**: PersonaPlex 7B (bfloat16)
- **VRAM**: ~14 GB minimum
- **Inference**: Real-time full-duplex speech-to-speech
- **Concurrency**: 1 session per GPU process (server uses `asyncio.Lock`)
- **Audio frame rate**: 12.5 fps (24kHz sample rate)

## GPU Sharing Math

One GPU has 24 hours/day. Each user consumes 2 hr/day:

- **Theoretical max users per GPU**: 24 hr / 2 hr = **12 users**
- **Realistic (with peak overlap)**: **8-10 users per GPU**

---

## Option 1: Budget GPU Cloud (Single GPU, Always-On)

| GPU                      | Provider  | $/hr        | $/month (always-on) | Users/GPU | Cost/user     | Margin     |
| ------------------------ | --------- | ----------- | -------------------- | --------- | ------------- | ---------- |
| RTX 3090 (24 GB)         | Vast.ai   | $0.15-0.20  | $108-144             | 10        | $10.80-14.40  | **28-46%** |
| RTX 4090 (24 GB)         | RunPod    | $0.35       | $252                 | 10        | $25.20        | **-26%** (loss) |
| A5000 (24 GB)            | Vast.ai   | $0.20-0.25  | $144-180             | 10        | $14.40-18.00  | **10-28%** |

**Verdict**: Margins too low. Single always-on GPUs cannot hit 80%+ at this price point.

---

## Option 2: Larger GPU, Multiple Concurrent Instances

An H100 80 GB or A100 80 GB can fit **4-5 concurrent model instances** (14 GB x 5 = 70 GB):

| GPU              | Provider  | $/hr        | $/month   | Instances | Users/GPU | Cost/user     | Margin          |
| ---------------- | --------- | ----------- | --------- | --------- | --------- | ------------- | --------------- |
| H100 80 GB       | Vast.ai   | $2.00       | $1,440    | 5         | 50        | $28.80        | **-44%** (loss) |
| A100 80 GB       | Vast.ai   | $1.00-1.20  | $720-864  | 4         | 40        | $18.00-21.60  | **-8 to 10%**   |
| A100 80 GB       | Lambda    | $0.80       | $576      | 4         | 40        | $14.40        | **28%**         |

**Verdict**: Better density but still far from 80%+ margin.

---

## Option 3: On-Demand / Pay-Per-Use (Spin Up During Calls Only)

Instead of always-on, **start a GPU instance only when a user initiates a call**:

| GPU                      | Provider             | $/hr        | Actual hrs/user/month | Cost/user   | Margin     |
| ------------------------ | -------------------- | ----------- | --------------------- | ----------- | ---------- |
| RTX 3090                 | Vast.ai              | $0.15       | 60                    | $9.00       | **55%**    |
| RTX 3090 (spot)          | Vast.ai              | $0.07-0.10  | 60                    | $4.20-6.00  | **70-79%** |
| T4 16 GB (preemptible)   | GCP                  | $0.11       | 60                    | $6.60       | **67%**    |

**Verdict**: Getting closer to target, but **cold start is 30-60 seconds** (loading 7B model into VRAM). Bad UX for real-time voice calls.

---

## Option 4: Hybrid Warm Pool (Recommended)

Keep a **minimum pool of warm GPUs** + **burst capacity for peaks**. This is the only realistic path to 80%+ margin.

### Assumptions

- **100 users** subscribed
- Not all users call simultaneously or use full 2 hr every day
- Real-world average usage: ~1 hr/day per active user
- Peak concurrency: ~15-20 simultaneous sessions
- 2 always-on GPUs handle base load (24 sessions/day each)
- 1 burst GPU for peak hours (~4 hr/day average)

### Cost Breakdown

| Component                                   | Cost/month |
| ------------------------------------------- | ---------- |
| 2x RTX 3090 always-on (Vast.ai @ $0.15/hr) | $216       |
| 1x RTX 3090 burst (~4 hr/day avg)           | $18        |
| **Total infrastructure**                    | **~$234**  |
| **Revenue (100 users x $20)**               | **$2,000** |
| **Margin**                                  | **~88%**   |

### Scaling Table

| Users | Always-On GPUs | Burst GPUs | Infra Cost | Revenue | Margin  |
| ----- | -------------- | ---------- | ---------- | ------- | ------- |
| 50    | 1              | 1          | $126       | $1,000  | **87%** |
| 100   | 2              | 1          | $234       | $2,000  | **88%** |
| 250   | 5              | 2          | $576       | $5,000  | **88%** |
| 500   | 10             | 3          | $1,098     | $10,000 | **89%** |

---

## Option 5: Self-Hosted Hardware

Purchase GPUs and colocate or run on-premises for maximum margin:

| Component                   | Cost             |
| --------------------------- | ---------------- |
| 2x Used RTX 3090            | $800-1,000 each (one-time) |
| Server (CPU, RAM, PSU)      | $500-800 (one-time)        |
| Colocation / power          | $50-100/month              |
| Internet (symmetric)        | $50-100/month              |
| **Monthly operating cost**  | **$100-200/month**         |
| Amortized hardware (12 mo)  | ~$175-235/month            |
| **Total (first year)**      | **$275-435/month**         |
| Revenue (100 users)         | $2,000/month               |
| **Margin (first year)**     | **78-86%**                 |
| **Margin (after payoff)**   | **90-95%**                 |

---

## Summary & Recommendation

| Strategy                        | Min Users | Margin    | Tradeoff                              |
| ------------------------------- | --------- | --------- | ------------------------------------- |
| Single always-on GPU            | 10        | 28-46%    | Simple but expensive per user         |
| Large GPU multi-instance        | 40        | 10-28%    | High fixed cost                       |
| On-demand spin-up               | Any       | 55-79%    | Cold start latency (bad UX)           |
| **Hybrid warm pool (Vast.ai)**  | **50+**   | **87-89%**| **Best balance of cost and UX**       |
| **Self-hosted hardware**        | **100+**  | **86-95%**| **Highest margin, upfront investment**|

### Action Items to Hit 80-85% Margin

1. **Scale to 50-100+ users** before GPU sharing becomes cost-efficient
2. **Use Vast.ai or self-hosted RTX 3090s** ($0.10-0.15/GPU-hr)
3. **Build a warm pool architecture** with session routing/load balancing
4. **Modify the server code** to remove the single-session `asyncio.Lock` and support multi-worker session routing
5. **Monitor real-world usage patterns** — actual average usage is likely below 2 hr/day, improving margins further

### Architecture Required

```
                         +------------------+
    Users (WebSocket) -> | Session Router / |
                         | Load Balancer    |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |             |             |
              +-----+-----+ +----+----+ +-----+-----+
              | GPU Worker | | GPU Worker| | GPU Worker |
              | (always-on)| | (always-on)| | (burst)   |
              | RTX 3090   | | RTX 3090  | | RTX 3090  |
              +------------+ +-----------+ +------------+
```

Each GPU worker runs an independent `moshi.server` instance. The session router assigns incoming calls to available workers and manages the warm pool lifecycle.
