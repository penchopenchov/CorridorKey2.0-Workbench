import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
import numpy as np
import torch
import corridorkey_launcher as launcher
from corridorkey.video_io import preprocess_frame_dir_path,probe_frame_dir_meta

ROOT=Path(__file__).resolve().parents[1]
FFMPEG=str(ROOT/'ffmpeg.exe') if (ROOT/'ffmpeg.exe').exists() else 'ffmpeg'

class StreamingTests(unittest.TestCase):
    def streaming(self):
        self.assertIsNotNone(importlib.util.find_spec('corridorkey.streaming'), 'float streaming decoder is missing')
        from corridorkey import streaming
        return streaming

    def test_float_decoder_matches_reference_exrs_for_transfers_and_ranges(self):
        stream=self.streaming()
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp' if (ROOT/'tmp').exists() else ROOT) as folder:
            folder=Path(folder)
            source=folder/'source.mp4'
            subprocess.run([FFMPEG,'-v','error','-y','-f','lavfi','-i','testsrc2=size=128x96:rate=24','-frames:v','3','-c:v','libx264','-pix_fmt','yuv420p',str(source)],check=True)
            for color in ('srgb','rec709','linear'):
                for levels in ('auto','full','limited'):
                    with self.subTest(color=color,levels=levels):
                        prepared=launcher.decode_video(source,folder/f'{color}_{levels}',color,levels,3,FFMPEG)
                        with stream.FloatVideoDecoder(source,width=128,height=96,count=3,ffmpeg_bin=FFMPEG,color=color,levels=levels) as decoder:
                            frames=list(decoder)
                        self.assertEqual(len(frames),3)
                        for index,frame in enumerate(frames):
                            expected=cv2.cvtColor(cv2.imread(str(prepared.path/f'frame_{index:08d}.exr'),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
                            np.testing.assert_array_equal(frame,expected)

    def test_streamed_preprocessing_matches_linear_exr_including_hdr_and_padding(self):
        stream=self.streaming()
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            folder=Path(folder)
            frame=np.random.default_rng(1).uniform(-.2,20,(73,101,3)).astype(np.float32)
            cv2.imwrite(str(folder/'000.exr'),frame[...,::-1])
            meta=probe_frame_dir_meta(str(folder),num_frames=-1,shape_quantum=128,fps=24,exr_linear=True,model_scale=1)
            expected=preprocess_frame_dir_path(str(folder/'000.exr'),meta,need_model=True)
            actual=stream.preprocess_linear_frame(frame,meta,need_model=True)
            for a,b in zip(actual,expected):
                self.assertTrue(torch.equal(a,b))

    def test_decoder_reports_truncated_source_and_closes(self):
        stream=self.streaming()
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            source=Path(folder)/'one.mp4'
            subprocess.run([FFMPEG,'-v','error','-y','-f','lavfi','-i','color=size=16x16','-frames:v','1',str(source)],check=True)
            decoder=stream.FloatVideoDecoder(source,width=16,height=16,count=2,ffmpeg_bin=FFMPEG)
            with self.assertRaisesRegex(RuntimeError,'frame|decode'):
                with decoder:
                    list(decoder)
            self.assertIsNotNone(decoder.proc.poll())

    def test_background_preparation_propagates_decode_error_and_stops_on_close(self):
        stream=self.streaming()
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            source=Path(folder)/'one.mp4'
            subprocess.run([FFMPEG,'-v','error','-y','-f','lavfi','-i','color=size=16x16','-frames:v','1',str(source)],check=True)
            meta={'source_hw':(16,16),'shape_hw':(128,128),'model_native_hw':(16,16),
                  'num_frames':2,'start_frame':0,'ffmpeg_bin':FFMPEG,'input_color':'srgb','input_range':'auto'}
            reader=stream.StreamingVideoWindowReader(source,meta)
            try:
                self.assertIsNotNone(getattr(reader,'prefetch',None),'bounded background preprocessing is missing')
                with self.assertRaisesRegex(RuntimeError,'frame|decode'):
                    reader.window(0,2)
            finally:
                reader.close()
            self.assertFalse(reader.prefetch.thread.is_alive())

    def test_cancelled_background_preparation_stops_with_a_full_queue(self):
        stream=self.streaming()
        self.assertTrue(hasattr(stream,'PrefetchedFrames'),'bounded background preprocessing is missing')
        # A finite real iterator fills a one-item queue; close must unblock put.
        class Decoder:
            def __iter__(self):
                for _ in range(100):
                    yield np.zeros((2,2,3),np.float32)
            def close(self):
                pass
        queue=stream.PrefetchedFrames(Decoder(),{'shape_hw':(2,2),'model_native_hw':(2,2)},size=1)
        next(queue)
        queue.close()
        self.assertFalse(queue.thread.is_alive())

    def test_decoder_failure_after_last_frame_cannot_report_success(self):
        stream=self.streaming()
        class Decoder:
            def __iter__(self):
                yield np.zeros((2,2,3),np.float32)
                raise RuntimeError('final decoder failure')
            def close(self):
                pass
        meta={'num_frames':1,'shape_hw':(2,2),'model_native_hw':(2,2)}
        with patch.object(stream,'open_decoder',return_value=Decoder()):
            reader=stream.StreamingVideoWindowReader('unused',meta)
            try:
                with self.assertRaisesRegex(RuntimeError,'final decoder failure'):
                    reader.window(0,1)
            finally:
                reader.close()

    def test_rvm_lookahead_does_not_evict_hann_overlap_or_irregular_tail(self):
        stream=self.streaming()
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            folder=Path(folder)
            for j in range(13):
                cv2.imwrite(str(folder/f'{j:03d}.exr'),np.full((16,16,3),j/13,np.float32))
            from corridorkey.video_io import SequentialFrameDirWindowReader
            meta=probe_frame_dir_meta(str(folder),num_frames=-1,shape_quantum=128,fps=24,exr_linear=True,model_scale=1)
            reader=SequentialFrameDirWindowReader(str(folder),meta)
            reader.retain_from=0
            first=reader.window(0,8)[0]
            reader.retain_from=4
            reader.window(4,12)
            reader.window(8,13)  # RVM's original next chunk.
            reader.retain_from=5
            tail=reader.window(5,13)[0]
            self.assertTrue(torch.equal(tail[:3],first[5:8]))

    def test_streamed_linear_adjustment_matches_explicitly_adjusted_exr(self):
        stream=self.streaming()
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            folder=Path(folder)
            frame=np.random.default_rng(3).uniform(-.1,3,(73,101,3)).astype(np.float32)
            adjusted=2*(.18+1.2*(frame-.18))
            cv2.imwrite(str(folder/'000.exr'),adjusted[...,::-1])
            meta=probe_frame_dir_meta(str(folder),num_frames=-1,shape_quantum=128,fps=24,exr_linear=True,model_scale=1)
            expected=preprocess_frame_dir_path(str(folder/'000.exr'),meta,need_model=True)
            meta.update(linear_brightness=2,linear_contrast=1.2)
            actual=stream.preprocess_linear_frame(frame,meta,need_model=True)
            for a,b in zip(actual,expected):
                self.assertTrue(torch.equal(a,b))

    def test_launcher_keeps_video_streaming_and_passes_explicit_color(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            folder=Path(folder)
            source=folder/'one.mp4'
            subprocess.run([FFMPEG,'-v','error','-y','-f','lavfi','-i','color=size=16x16','-frames:v','1',str(source)],check=True)
            settings=launcher.Settings(input_color='rec709',input_range='limited',cache_root=folder/'cache')
            with launcher.prepare_job(launcher.Job(source,source),settings,FFMPEG) as (media,hint,cache):
                self.assertEqual(media.path,source)
                command=launcher.inference_command(media,hint,folder/'out',settings,FFMPEG,cache)
                self.assertEqual(command[command.index('--input_color')+1],'rec709')
                self.assertEqual(command[command.index('--input_range')+1],'limited')
                self.assertEqual(command[command.index('--video_decoder')+1],'ffmpeg')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_gpu_preprocessing_preserves_hdr_and_reflect_padding_with_bf16_tolerance(self):
        stream=self.streaming()
        self.assertTrue(hasattr(stream,'preprocess_linear_frame_cuda'),'GPU preprocessing is missing')
        for hw in ((73,101),(1,1),(216,384)):
            frame=np.random.default_rng(2).uniform(-.2,20,(*hw,3)).astype(np.float32)
            meta={'shape_hw':tuple(((x+127)//128)*128 for x in hw),'model_native_hw':hw}
            cpu=stream.preprocess_linear_frame(frame,meta,need_model=True)
            gpu=stream.preprocess_linear_frame_cuda(frame,meta,need_model=True)
            for a,b in zip(cpu,gpu):
                self.assertTrue(bool(b.isfinite().all()))
                torch.testing.assert_close(a,b.cpu(),rtol=0,atol=.0078125)

if __name__=='__main__':
    unittest.main()
