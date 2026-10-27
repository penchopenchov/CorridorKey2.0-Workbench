import inspect
import unittest
import torch
import infer
from corridorkey.blocks import SwiGLU
from corridorkey.runtime import InferenceConfig, configure_low_vram_inference

class BatchTests(unittest.TestCase):
    def test_normal_job_clears_low_memory_chunking_from_previous_job(self):
        model=torch.nn.Module()
        model.stage1=torch.nn.Linear(2,2)
        model.mlp=SwiGLU(8)
        configure_low_vram_inference(model,InferenceConfig(low_vram=True))
        self.assertGreater(model.mlp.inference_chunk_tokens,0)
        configure_low_vram_inference(model,InferenceConfig(low_vram=False))
        self.assertEqual(model.mlp.inference_chunk_tokens,0)
        self.assertEqual(model.stage1.low_vram_enc_frame_chunk,0)

    def test_cli_exposes_reusable_model_entrypoint(self):
        self.assertIn('model',inspect.signature(infer.main).parameters)
        self.assertIn('argv',inspect.signature(infer.main).parameters)

if __name__=='__main__':
    unittest.main()
