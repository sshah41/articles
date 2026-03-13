# Architecture Deep Dive: Nemotron 3 Super
## How NVIDIA’s Hybrid MoE Unlocks 1M-Token Agentic AI

**By [Author Name]**  
*Released March 2026 · Audience: ML engineers, technical PMs, and technically-minded AI enthusiasts*

Nemotron 3 Super represents one of the most significant architectural leaps in open-weight large language models for agentic and long-context workloads. Released by NVIDIA in March 2026, this model features roughly 120B total parameters, with about 12–13B active per forward pass. This Mixture-of-Experts (MoE) model fuses three co-designed innovations — a hybrid Mamba-Transformer backbone, LatentMoE, and multi-token prediction (MTP) — to make 1M-token contexts substantially more plausible for self-hosted open-weight systems.

The result is a model that provides a strong new open-model reference point for long-context agents, multi-turn research assistants, and RAG pipelines over entire codebases or books. This article unpacks why — starting with the memory and bandwidth scaling constraints that have limited every major Transformer-based model before it.

---

### The Problem: KV Cache Explosion at Scale

Many strong open models — the Qwen series, Llama-4 derivatives, DeepSeek, and GPT-OSS equivalents — excel at short-to-medium contexts (4k–128k tokens) but degrade sharply, or become outright unusable, when pushed toward 256k+, 512k+, or 1M-token regimes. To understand why this **1-million token context window** remains a challenge for agentic workflows, we need to abandon abstract descriptions and look at decode-time cache growth: the physics of the KV cache explosion.

When feeding legacy large language models a massive prompt, the network is forced to compute and rigorously store Key and Value tensors for every token, across *all* attention layers. This mechanism inherently creates a continuously expanding 4-dimensional tensor, the KV cache. The memory volume of this tensor is strictly defined as:

Volume = 2 * L_attn * (B * S * N_h * d_h) * p

Where B is Batch Size, S is Sequence length, N_h is the number of heads, d_h is the head dimension, and p represents precision bytes (typically 2 for FP16 or BF16).

Let's apply this to real-world models in BF16 at 1M tokens (Author Estimates based on typical GQA configurations):

| Model | Attn Layers / KV Heads | KV Cache (BF16) at 1M Tokens |
| :--- | :--- | :--- |
| **Qwen3.5-122B-A10B** | ~12–16 layers, 8 KV heads | ~24 GB |
| **Llama-3 70B (typical)** | 30–40 layers, 8 KV heads | ~32–40 GB |
| **Generic 100B+ MoE** | Varies, large GQA | ~20–30 GB |
| ✨ **Nemotron 3 Super** | Few strategic attention layers, 2 KV heads | **Markedly smaller KV cache (author estimate)** |

Even at 262k tokens — a realistic long-document or multi-turn agent session — a Qwen-class model already consumes ~6 GB of VRAM for the KV cache alone. At 1M tokens, that balloons to ~24 GB: most of an H100/B200's VRAM budget before accounting for weights and activations. Inference latency also balloons, as generation is memory-bandwidth bound: the system spends most of its time reading and writing enormous KV tensors, not computing.

![Context Length vs. VRAM Usage](/Users/syednabeelshah/.gemini/antigravity/brain/bce371be-b4b2-4985-a7ed-7c8e306736e1/kv_cache_scientific_1773336832557.png)
*Figure 1: Context Length vs. VRAM Usage. Transformer-based long-context inference faces both quadratic full-sequence attention costs and linear KV-cache growth during autoregressive decoding. Nemotron reduces decode-time cache pressure by using only a small number of attention layers with 2 KV heads.*

Nemotron 3 Super reduces this attention-state overhead by drastically reducing the scalar multiplier: L_attn. Rather than computing attention everywhere, Nemotron uses a relatively small number of strategically placed **Grouped-Query Attention (GQA)** layers acting as sparse "global anchors." This shrinks the model's overall T_KV footprint significantly.

At 1M tokens, Nemotron saves roughly 18 GB of VRAM compared to a dense Qwen3.5-122B-class model. That’s enough headroom to run larger batch sizes, stack longer reasoning chains in agentic loops, or operate multiple independent agents concurrently on a single GPU.

---

### The Hybrid Architecture: An 88-Layer Stack

If only a few layers handle the heavily associative O(N²) scalar dot products of Attention, what handles the bulk of the sequence processing? Nemotron flips the equation by utilizing architectural restraint: use attention sparingly for precise long-range retrieval, and handle the vast majority of sequence processing with state-space layers.

NVIDIA describes Nemotron 3 Super as an 88-layer hybrid stack that interleaves Mamba-2 blocks, sparse LatentMoE layers, and a small number of grouped-query attention layers (with 32 query heads and 2 KV heads) used as global retrieval anchors.

![Architectural block diagram showing sequential Mamba-2 blocks interspersed with sparse Latent MoE routing mechanisms.](/Users/syednabeelshah/.gemini/antigravity/brain/bce371be-b4b2-4985-a7ed-7c8e306736e1/architecture_block_scientific_1773336847584.png)
*Figure 2: Architectural Stack Diagram showing the Mamba-2 bidirectional scan foundation interspersed with Latent MoE routing.*

Because Mamba-2 is fundamentally a State-Space Model (SSM), it operates completely differently from attention. Rather than looking back at the entire history of tokens, it recursively updates a fixed-size hidden state at each timestep:

h(t) = A * h(t-1) + B * x(t)

Because the spatial volume of the hidden state remains strictly constant regardless of how many tokens are ingested, the Mamba-2 layers do not contribute to Transformer-style KV-cache growth. They process history continuously in linear O(N) time.

---

### Three Co-Designed Innovations

The KV cache efficiency isn’t the whole story. Nemotron 3 Super combines three innovations that together produce a model that is simultaneously faster, more accurate, and cheaper to run than its size class would suggest.

#### 1. LatentMoE: Dimensionality Bottlenecks

Interleaved throughout this sparse architecture are 23 **Latent Mixture-of-Experts (LatentMoE)** layers. Standard MoE layers route tokens to a subset of experts to reduce FLOPs, handling high-dimensional vectors directly. 

LatentMoE takes a more computationally elegant approach. It explicitly multiplies the input vector x by a specific down-projection matrix, W_down, to physically compress it into a continuous latent bottleneck:

z = W_down * x

Once inside this lower-dimensional bottleneck space, the tokens are rapidly routed. NVIDIA’s architecture table lists 512 experts per layer with top-k = 22, alongside a shared-expert component. The output is then expanded back to full dimensionality via W_up. The mathematical result? Vastly richer geometric feature combinations at roughly the cost of activating far fewer experts in a standard design, translating to better accuracy per FLOP.

#### 2. Multi-Token Prediction (MTP): Faster Inference Natively

Rather than predicting one token at a time, MTP trains the model to predict multiple future tokens simultaneously. At inference time this enables **speculative decoding** — supporting native speculative decoding and materially faster inference; NVIDIA reports up to 3× faster inference from MTP in public materials. Unlike bolted-on speculative decoding implementations, MTP is native to Nemotron 3 Super’s training objective, making it reliable across diverse tasks and long reasoning chains.

#### 3. NVFP4 Native Training: Precision Quantization

Memory capacity is critical, but large model inference is ultimately bounded by **memory bandwidth**—how fast the GPU can shovel weights from HBM (High Bandwidth Memory) into the SRAM processing cores.

To maximize throughput, Nemotron 3 Super uses **NVFP4** aggressively where it helps most, within a mixed-precision training recipe designed to preserve stability and accuracy. It is trained natively on 25 trillion tokens using NVIDIA's 4-bit floating point format for many components, from the very first gradient update.

![Schematic representation of extreme parameter compression from 16-bit to NVFP4 formats.](/Users/syednabeelshah/.gemini/antigravity/brain/bce371be-b4b2-4985-a7ed-7c8e306736e1/nvfp4_quantization_scientific_1773336860703.png)
*Figure 3: Schematic representation mapping the compression of large dynamic range FP16 parameters into the dense NVFP4 format.*

Post-hoc quantization of large models typically incurs accuracy degradation that requires careful calibration to recover. By contrast, NVFP4-native pretraining lets the model’s weights adapt to reduced precision throughout training. NVIDIA argues that this mixed-precision recipe preserves competitive model quality while substantially improving efficiency. By shrinking the precision scalar p for many weights, data density skyrockets.

---

### Throughput: The Numbers in Context

NVIDIA’s benchmarks report the massive real-world impact of these combined mathematical optimizations:

| Comparison | Throughput Gain |
| :--- | :--- |
| **vs. GPT-OSS-120B (Transformer MoE)** | 2.2× higher throughput |
| **vs. Qwen3.5-122B-A10B** | **7.5× higher throughput** |
| **vs. 8-bit models on Hopper (on Blackwell)** | 4× faster inference |
| **vs. previous Nemotron Super model** | Up to 5× higher throughput, 2× higher accuracy |

The 7.5× gain over Qwen3.5-122B is particularly striking because both models share a broadly similar active parameter count (~12–13B active). The difference is almost entirely architectural: Qwen’s Transformer-based attention must read and write a massive KV cache on every decoding step; Nemotron’s Mamba-2 layers recur over a fixed-size state, and its attention layers touch a significantly smaller cache.

NVIDIA says Nemotron 3 Super powers its AI-Q research agent, which reached No. 1 on DeepResearch Bench and DeepResearch Bench II, confirming that architecture-level throughput gains are achieved while remaining highly competitive on the reported benchmarks.

---

### Why This Matters for Real Agentic Workloads

Most production agent systems today are constrained to the 8k–128k token range. Beyond that, three forces conspire against them: VRAM exhaustion prevents loading the full context; latency explodes as agents spend more time managing memory than reasoning; and cost compounds because longer contexts require either more GPUs or expensive cloud inference time.

Nemotron 3 Super changes each of these constraints simultaneously. NVIDIA has made 1M-token operation substantially more plausible for self-hosted open-weight systems, though deployment at that context length still depends heavily on runtime configuration and memory budget. Multi-agent collaboration — such as a code review agent, a testing agent, and a documentation agent sharing a common context window — becomes far more feasible. And the MTP-accelerated throughput means that even during long chain-of-thought reasoning traces, the model's generation is much faster.

**What Opens Up at 1M Context:**
*   RAG over much larger codebases or book-length documents with less aggressive chunking.
*   Multi-agent workflows sharing a full long-term memory window.
*   Multi-turn research assistants with months of conversation history.
*   Code generation with full repository context in a single pass.
*   Legal or compliance review over very large contract collections in fewer passes.

The base model was pretrained on 25 trillion total tokens. NVIDIA also describes a broader post-training pipeline involving supervised fine-tuning and reinforcement learning for agentic behavior. This means the model has been explicitly optimized for agentic decision-making, not just long-context retrieval.

### The Bigger Architectural Signal

By uniquely composing sparse A(Q,K,V) for recall, recursive A*h Mamba states for sequence flow, and dense 4-bit W matrices for bandwidth throughput, Nemotron 3 Super is not simply a larger or better-trained Transformer. It represents a deliberate architectural thesis.

**Transformer attention-everywhere is the wrong default for long-context reasoning.**

The hybrid Mamba-Transformer approach inverts this assumption: Mamba handles the default case (sequence processing with linear complexity), and attention is reserved for the cases where it uniquely excels (precise token retrieval, long-range associative recall). The result is a model that degrades gracefully at scale rather than catastrophically.

With open weights, full training recipes, and open datasets under the NVIDIA Open Model License, NVIDIA has demonstrated a highly effective approach to mitigating the memory wall. For anyone building autonomous agents, long-document RAG pipelines, or multi-turn research assistants, this architecture represents a strong new open-model reference point.
