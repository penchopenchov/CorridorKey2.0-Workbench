import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

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

if __name__=='__main__':
    unittest.main()
