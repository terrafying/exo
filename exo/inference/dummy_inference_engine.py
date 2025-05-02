from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.inference.tokenizers import DummyTokenizer

class DummyInferenceEngine(InferenceEngine):
  def __init__(self):
    self.shard = None
    self.vocab_size = 1000
    self.hidden_size = 256
    self.eos_token_id = 0
    self.latency_mean = 0.1
    self.latency_stddev = 0.02
    self.num_generate_dummy_tokens = 10
    self.tokenizer = DummyTokenizer()

  async def encode(self, request_id: str, shard: Shard, prompt: str) -> np.ndarray:
    """Encode a prompt into tokens"""
    await self.ensure_shard(shard)
    tokens = [ord(c) for c in prompt]
    return np.array(tokens).reshape(1, -1)  # Ensure 2D output

  async def sample(self, request_id: str, shard: Shard, logits: np.ndarray) -> np.ndarray:
    """Sample from logits"""
    await self.ensure_shard(shard)
    if logits.shape[-1] == 0:
      return np.array([[0]])  # Return EOS token if no logits
    return np.array([[np.argmax(logits[-1])]]).reshape(1, -1)  # Ensure 2D output

  async def decode(self, request_id: str, shard: Shard, tokens: np.ndarray) -> str:
    """Decode tokens into text"""
    await self.ensure_shard(shard)
    if tokens.ndim == 2:
      tokens = tokens.flatten()  # Flatten if 2D
    return ''.join(chr(int(t)) for t in tokens)

  async def infer_tensor(self, request_id: str, shard: Shard, data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    """Infer tensor through the model"""
    await self.ensure_shard(shard)
    
    # Ensure input is 2D
    if data.ndim == 1:
      data = data.reshape(1, -1)
      
    # Add 1 to each element if this is the last layer
    if self.shard.is_last_layer():
      data = data + 1
      
    return data, inference_state or {}

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard: return
    self.shard = shard
  
  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
