import unittest
import numpy as np
import torch
import threading
from corridorkey.runtime import _srgb_to_bgr_uint16_cpu, _srgb_to_bgra_uint16_cpu

class OutputTests(unittest.TestCase):
    def test_failed_encoder_does_not_deadlock_a_full_queue_or_release(self):
        from corridorkey.video_io import ThreadedVideoWriter,WriterConfig
        class Sink:
            def write(self,data):
                raise BrokenPipeError('injected encoder failure')
            def close(self):
                raise BrokenPipeError('injected close failure')
        class Process:
            stdin=Sink()
            waited=False
            def wait(self):
                self.waited=True
                return 1
        class FailedWriter(ThreadedVideoWriter):
            def _start(self):
                self.proc=Process()
                self.thread.start()
        writer=FailedWriter('unused',24,(2,2),WriterConfig(),maxsize=1)
        errors=[]
        def produce():
            try:
                for _ in range(10):
                    writer.write_many(np.zeros((1,2,2,3),np.uint16))
            except BrokenPipeError as error:
                errors.append(str(error))
        producer=threading.Thread(target=produce,daemon=True)
        producer.start()
        producer.join(timeout=3)
        self.assertFalse(producer.is_alive(),'producer deadlocked after encoder failure')
        self.assertEqual(errors,['injected encoder failure'])
        with self.assertRaisesRegex(BrokenPipeError,'injected'):
            writer.release()
        self.assertFalse(writer.thread.is_alive())
        self.assertTrue(writer.proc.waited,'failed encoder process was not reaped')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_video_download_matches_reference_truncation_and_channel_order(self):
        values=np.random.default_rng(7).uniform(-.1,1.1,(2,3,37,49)).astype(np.float32)
        values.flat[:6]=[0.,1.,.5,1/65535,.5/65535,65534.5/65535]
        x=torch.from_numpy(values).cuda()
        result=_srgb_to_bgr_uint16_cpu(x)
        self.assertTrue(hasattr(result,'resolve'),'video download still blocks the inference thread')
        expected=(np.clip(values,0,1)[:,[2,1,0]].transpose(0,2,3,1)*65535).astype(np.uint16)
        np.testing.assert_array_equal(result.resolve(),expected)
        a=x[:,0:1]
        rgba=_srgb_to_bgra_uint16_cpu(x,a).resolve()
        expected_a=(np.clip(values[:,0:1],0,1).transpose(0,2,3,1)*65535).astype(np.uint16)
        np.testing.assert_array_equal(rgba,np.concatenate([expected,expected_a],axis=-1))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_alpha_download_uses_one_channel_without_changing_rgb_bytes(self):
        from corridorkey import video_io
        x=torch.rand(2,19,23,1,device='cuda')
        import inspect
        self.assertIn('expand_channels',inspect.signature(video_io.download_uint16).parameters)
        result=video_io.download_uint16(x,expand_channels=3)
        self.assertEqual(result.array.shape[-1],1)
        expected=np.repeat((x.cpu().numpy()*65535).astype(np.uint16),3,axis=-1)
        np.testing.assert_array_equal(result.resolve(),expected)

if __name__=='__main__':
    unittest.main()
