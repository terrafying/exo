from exo.inference.inference_engine import InferenceEngine
import transformers
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
import numpy as np
import torch
import asyncio
from loguru import logger
import torch
from concurrent.futures import ThreadPoolExecutor

TEMPERATURE = 0.85
def download_model_from_model_id(model_id):
    pass

_executor = ThreadPoolExecutor(max_workers=1)
class HuggingfaceInferenceEngine(InferenceEngine):
    def __init__(self, shard: Shard):
        self.shard = None
        self.model_id = shard.model_id
        self.executor = _executor
        self.model = None
        self.tokenizer = None
    
    async def encode(self, shard, prompt):
        """Encodes prompt to tokens using the tokenizer"""
        await self.ensure_shard(shard)
        tokens = self.tokenizer(prompt, return_tensors="pt")
        return tokens
    
    async def sample(self, x: np.ndarray, temp=TEMPERATURE, top_p: float = 0.0) -> np.ndarray:
        """Samples the next token from the logits"""
        def sample_wrapper():
            logits = torch.Tensor(x[:, -1, :])
            
            processors_list = [
                transformers.TemperatureLogitsWarper(temp),
            ]
            if top_p > 0:
                processors_list.append(transformers.TopPLogitsWarper(top_p))
                
            processors = transformers.LogitsProcessorList(processors_list)
            filtered_logits = processors(None, logits.detach().cpu().numpy())
            probs = torch.nn.functional.softmax(torch.Tensor(filtered_logits), dim=-1)
            return probs.multinomial(num_samples=1).cpu().numpy().astype(int)
        
        return await asyncio.get_running_loop().run_in_executor(self.executor, sample_wrapper)
    
    async def decode(self, shard, tokens):
        """Decodes tokens to text using the tokenizer"""
        await self.ensure_shard(shard)
        return self.tokenizer.decode(tokens)
    
    async def infer_tensor(self, request_id, shard, input_data, inference_state: dict = None):
        await self.ensure_shard(shard)
        logger.info(f"Running inference for request {request_id}, shard {shard}")
        
        if inference_state is None:
            inference_state = {}
        
        # Handle input data
        if isinstance(input_data, dict):
            input_ids = input_data["input_ids"]
            attention_mask = input_data.get("attention_mask")
        else:
            input_ids = torch.tensor(input_data) if isinstance(input_data, np.ndarray) else input_data
            attention_mask = None
            
        # Ensure input_ids is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        # Generate position IDs
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Create causal mask
        causal_mask = self._prepare_causal_mask(
            input_ids.shape[0],
            seq_length,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        
        # Initial embedding
        if shard.start_layer == 0:
            outputs = self.model.model.embed_tokens(input_ids)
        else:
            outputs = input_ids
        
        # Process through layers
        for i, layer_idx in enumerate(range(shard.start_layer, shard.end_layer + 1)):
            residual = outputs
            
            # Get position embeddings for RoPE
            cos, sin = self.model.model.layers[i].self_attn.rotary_emb(position_ids)
            
            outputs = self.model.model.layers[i](
                hidden_states=outputs,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=(cos, sin)
            )
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            outputs = outputs + residual
        
        if shard.end_layer == shard.n_layers - 1:
            outputs = self.model.model.norm(outputs)
            outputs = self.model.lm_head(outputs)
            
        return outputs, inference_state

    def _prepare_causal_mask(self, batch_size, seq_length, dtype, device):
        mask = torch.triu(torch.ones((seq_length, seq_length), device=device) * -float("inf"), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, 1, seq_length, seq_length)
        return mask.to(dtype=dtype)
    
    async def ensure_shard(self, shard: Shard):
        if self.shard == shard:
            return
        
        logger.info(f"Downloading model {shard.model_id}")
        self.shard = shard
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        logger.info(f"Model {shard.model_id} downloaded")

    async def infer_prompt(self, request_id, shard, prompt, inference_state = None):
        """takes prompt and infers till the end layer of the shard"""
        ## encoding only needed when prompt is given
        tokens = await self.encode(shard, prompt) ## [input_ids, attention_mask, position_ids]
        # x = tokens.reshape(1, -1)
        output_data, inference_state = await self.infer_tensor(request_id, shard, tokens, inference_state)
        return output_data, inference_state

    async def load_checkpoint(self, shard: Shard, path: str):
        """Load a model checkpoint from the specified path"""
        await self.ensure_shard(shard)
        try:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(path)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
            logger.info(f"Loaded checkpoint from {path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint from {path}: {e}")
            raise