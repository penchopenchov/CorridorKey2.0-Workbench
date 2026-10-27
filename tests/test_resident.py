import importlib.util
import unittest
import numpy as np
import torch

class ResidentTests(unittest.TestCase):
    def test_reference_quality_default_does_not_change_foundation_batch_shapes(self):
        from corridorkey.runtime import InferenceConfig
        self.assertFalse(InferenceConfig().cache_foundations)

    def test_foundation_cache_extracts_new_frames_once_and_evicts_old_frames(self):
        from corridorkey import resident
        self.assertTrue(hasattr(resident,'FoundationCache'),'foundation cache is missing')
        cache=resident.FoundationCache()
        batches=[]
        def extract(frames):
            batches.append(len(frames))
            return {'moge':frames*2,'cradio':frames+1}
        first=torch.arange(8).view(8,1,1,1).float()
        second=torch.arange(4,12).view(8,1,1,1).float()
        cache.window(0,first,extract)
        out=cache.window(4,second,extract)
        self.assertEqual(batches,[8,4])
        self.assertTrue(torch.equal(out['moge'],second*2))
        self.assertEqual(set(cache.frames),set(range(4,12)))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_hann_accumulation_preserves_numpy_float16_boundary_and_tail_order(self):
        self.assertIsNotNone(importlib.util.find_spec('corridorkey.resident'),'resident accumulator is missing')
        from corridorkey.resident import BatonAccumulator
        from corridorkey.video_io import taper_weights,window_starts
        acc=BatonAccumulator()
        reference={}
        weights={}
        emitted=[]
        starts=window_starts(13,8,4)
        for i,t0 in enumerate(starts):
            x=torch.randn(8,5,17,19,device='cuda')
            w=taper_weights(8,4,is_first=i==0,is_last=i==len(starts)-1)
            acc.add(t0,x,w)
            weighted=(x*w.to('cuda').view(8,1,1,1)).cpu().numpy()
            for j in range(8):
                k=t0+j
                reference[k]=reference[k]+weighted[j] if k in reference else weighted[j].copy()
                weights[k]=weights.get(k,0.)+float(w[j])
            limit=starts[i+1] if i+1<len(starts) else 14
            for k,value in acc.flush_before(limit):
                np.testing.assert_array_equal(value.cpu().numpy(),(reference.pop(k)/weights.pop(k)).astype(np.float16))
                emitted.append(k)
        self.assertEqual(emitted,list(range(13)))
        self.assertEqual(len(acc),0)

if __name__=='__main__':
    unittest.main()
