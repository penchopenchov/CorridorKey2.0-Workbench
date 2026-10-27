import unittest
import torch
from unittest.mock import patch
from contextlib import contextmanager
from corridorkey import runtime

class BackendTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_reference_predictions_return_after_cudnn_and_native_options(self):
        class Attention(torch.nn.Module):
            def forward(self,rgb,baton):
                return torch.nn.functional.scaled_dot_product_attention(rgb,baton,baton)
        model=torch.nn.Module()
        model.stage2=Attention()
        q=torch.randn(2,2,16,80,device='cuda',dtype=torch.bfloat16)
        k=torch.randn_like(q)
        with torch.inference_mode():
            before=runtime._stage2_forward(model,q,k,runtime.InferenceConfig())
            faster=runtime._stage2_forward(model,q,k,runtime.InferenceConfig(
                stage2_attention='cudnn',stage2_native_convolutions=True))
            after=runtime._stage2_forward(model,q,k,runtime.InferenceConfig())
        self.assertTrue(bool(faster.isfinite().all()))
        self.assertTrue(torch.equal(before,after))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cudnn_preference_falls_back_for_unsupported_head_dimension(self):
        class Attention(torch.nn.Module):
            def forward(self,rgb,baton):
                return torch.nn.functional.scaled_dot_product_attention(rgb,baton,baton)
        model=torch.nn.Module()
        model.stage2=Attention()
        q=torch.randn(1,2,11,7,device='cuda',dtype=torch.bfloat16)
        k=torch.randn_like(q)
        with torch.inference_mode():
            actual=runtime._stage2_forward(model,q,k,runtime.InferenceConfig(stage2_attention='cudnn'))
        self.assertEqual(actual.shape,q.shape)
        self.assertTrue(bool(actual.isfinite().all()))

    def test_cudnn_preference_keeps_compatible_fallbacks_and_restores_flags(self):
        observed=[]
        class Add(torch.nn.Module):
            def forward(self,rgb,baton):
                observed.append((torch.backends.cuda.cudnn_sdp_enabled(),
                                 torch.backends.cuda.mem_efficient_sdp_enabled(),
                                 torch.backends.cuda.math_sdp_enabled(),torch.backends.cudnn.enabled))
                return rgb+baton
        model=torch.nn.Module()
        model.stage2=Add()
        x=torch.ones(1,1,3,2,2)
        cfg=runtime.InferenceConfig(stage2_attention='cudnn')
        cfg.stage2_native_convolutions=True
        previous=(torch.backends.cudnn.enabled,torch.backends.cudnn.benchmark)
        actual=runtime._stage2_forward(model,x,x,cfg)
        self.assertTrue(torch.equal(actual,x+x))
        self.assertEqual(observed,[(True,True,True,False)])
        self.assertEqual((torch.backends.cudnn.enabled,torch.backends.cudnn.benchmark),previous)

    def test_compiled_execution_fallback_retains_requested_attention(self):
        state={'active':False,'seen':[]}
        @contextmanager
        def selected(*args,**kwargs):
            state['active']=True
            try:
                yield
            finally:
                state['active']=False
        class Add(torch.nn.Module):
            def forward(self,rgb,baton):
                state['seen'].append(state['active'])
                return rgb+baton
        def broken(*args,**kwargs):
            raise RuntimeError('injected compiled execution failure')
        model=torch.nn.Module()
        model.stage2=Add()
        x=torch.ones(1,1,3,2,2)
        cfg=runtime.InferenceConfig(compile_stage2='default',stage2_attention='efficient')
        with patch.object(runtime.importlib.util,'find_spec',return_value=object()), patch.object(torch,'compile',return_value=broken), patch('torch.nn.attention.sdpa_kernel',selected):
            with self.assertWarnsRegex(UserWarning,'eager'):
                runtime._stage2_forward(model,x,x,cfg)
            runtime._stage2_forward(model,x,x,cfg)
        self.assertEqual(state['seen'],[True,True])

    def test_compiler_construction_failure_falls_back_to_eager(self):
        class Add(torch.nn.Module):
            def forward(self,rgb,baton):
                return rgb+baton
        model=torch.nn.Module()
        model.stage2=Add()
        x=torch.ones(1,1,3,2,2)
        cfg=runtime.InferenceConfig(compile_stage2='default')
        with patch.object(runtime.importlib.util,'find_spec',return_value=object()), patch.object(torch,'compile',side_effect=RuntimeError('compiler initialization failed')):
            with self.assertWarnsRegex(UserWarning,'eager'):
                actual=runtime._stage2_forward(model,x,x,cfg)
        self.assertTrue(torch.equal(actual,x+x))

    def test_default_backend_preserves_eager_execution(self):
        cfg=runtime.InferenceConfig()
        self.assertEqual(getattr(cfg,'stage2_attention',None),'default')
        self.assertEqual(getattr(cfg,'compile_stage2',None),'none')

    def test_resident_memory_estimate_reserves_space_for_gpu_buffers(self):
        baseline=runtime.select_vram_profile((2176,3840),240,16)
        import inspect
        self.assertIn('resident_buffers',inspect.signature(runtime.select_vram_profile).parameters)
        resident=runtime.select_vram_profile((2176,3840),240,16,resident_buffers=True)
        self.assertFalse(baseline[2])
        self.assertTrue(resident[2])

if __name__=='__main__':
    unittest.main()
