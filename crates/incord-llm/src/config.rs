/// Incord model configuration (OpenBMB MiniCPM5-2B / LlamaForCausalLM architecture).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullAttn = 0,
    DeltaNet = 1,
}

#[derive(Debug, Clone)]
pub struct IncordConfig {
    // Language model
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attn_heads: usize,    // full attention Q heads
    pub num_kv_heads: usize,      // full attention KV heads
    pub attn_head_dim: usize,     // full attention head dim
    pub num_qk_heads: usize,      // DeltaNet QK heads (0 for pure LLaMA)
    pub num_v_heads: usize,       // DeltaNet V heads (0 for pure LLaMA)
    pub deltanet_head_dim: usize, // DeltaNet head dim (0 for pure LLaMA)
    pub conv_kernel_size: usize,  // DeltaNet conv1d kernel (0 for pure LLaMA)
    pub intermediate_size: usize,
    pub rope_theta: f64,
    pub partial_rotary_factor: f32, // fraction of head_dim that gets RoPE (1.0 for MiniCPM)
    pub rms_norm_eps: f64,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub layer_types: Vec<LayerType>,
    // Vision (optional)
    pub has_vision: bool,
    pub vision_hidden: usize,
    pub vision_layers: usize,
    pub vision_heads: usize,
    pub vision_ffn: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
}

pub type IncoreConfig = IncordConfig;
pub type Qor4bConfig = IncordConfig;

impl IncordConfig {
    /// Default configuration for OpenBMB MiniCPM5-2B.
    /// Pure LlamaForCausalLM Transformer architecture:
    /// - 42 layers (all FullAttn)
    /// - 2048 hidden size
    /// - 6144 intermediate size (SwiGLU)
    /// - 16 attention heads (head_dim = 128)
    /// - 2 key-value heads (GQA with 8:1 ratio)
    /// - 130560 vocabulary size
    /// - 5,000,000.0 RoPE theta (128K context)
    /// - 1e-6 RMSNorm epsilon
    pub fn default_2b() -> Self {
        Self {
            vocab_size: 130_560,
            hidden_size: 2048,
            num_layers: 42,
            num_attn_heads: 16,
            num_kv_heads: 2,
            attn_head_dim: 128,
            num_qk_heads: 0,
            num_v_heads: 0,
            deltanet_head_dim: 0,
            conv_kernel_size: 0,
            intermediate_size: 6144,
            rope_theta: 5_000_000.0,
            partial_rotary_factor: 1.0,
            rms_norm_eps: 1e-6,
            eos_token_id: 1, // OpenBMB MiniCPM5-2B: [1, 130073]
            bos_token_id: 0,
            max_position_embeddings: 131_072,
            tie_word_embeddings: false,
            layer_types: vec![LayerType::FullAttn; 42],
            has_vision: false,
            vision_hidden: 0,
            vision_layers: 0,
            vision_heads: 0,
            vision_ffn: 0,
            patch_size: 0,
            spatial_merge_size: 0,
            num_position_embeddings: 0,
        }
    }

    /// Alias for default config.
    pub fn default_4b() -> Self {
        Self::default_2b()
    }

    /// Number of KV groups for GQA in full attention layers.
    pub fn num_kv_groups(&self) -> usize {
        if self.num_kv_heads == 0 {
            1
        } else {
            self.num_attn_heads / self.num_kv_heads
        }
    }

    /// Rotary dimension for full attention (partial_rotary_factor * attn_head_dim).
    pub fn rope_dim(&self) -> usize {
        (self.attn_head_dim as f32 * self.partial_rotary_factor) as usize
    }

    /// Full attention Q dimension.
    pub fn attn_q_dim(&self) -> usize {
        self.num_attn_heads * self.attn_head_dim
    }

    /// Full attention KV dimension.
    pub fn attn_kv_dim(&self) -> usize {
        self.num_kv_heads * self.attn_head_dim
    }

    /// DeltaNet QKV projection output size: Q + K + V.
    pub fn deltanet_qkv_dim(&self) -> usize {
        let q = self.num_qk_heads * self.deltanet_head_dim;
        let k = self.num_qk_heads * self.deltanet_head_dim;
        let v = self.num_v_heads * self.deltanet_head_dim;
        q + k + v
    }

    /// DeltaNet Q dimension.
    pub fn deltanet_q_dim(&self) -> usize {
        self.num_qk_heads * self.deltanet_head_dim
    }

    /// DeltaNet K dimension (same as Q).
    pub fn deltanet_k_dim(&self) -> usize {
        self.num_qk_heads * self.deltanet_head_dim
    }

    /// DeltaNet V dimension.
    pub fn deltanet_v_dim(&self) -> usize {
        self.num_v_heads * self.deltanet_head_dim
    }

    /// Vision head dimension.
    pub fn vision_head_dim(&self) -> usize {
        if self.vision_heads == 0 {
            0
        } else {
            self.vision_hidden / self.vision_heads
        }
    }
}
