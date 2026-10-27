"""Exercise the real guided-command -> argparse -> runtime-config boundary."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import infer
import corridorkey_launcher as launcher
import run_interactive
from corridorkey import runtime

ROOT=Path(__file__).resolve().parents[1]

class CliSelectionTests(unittest.TestCase):
    def test_legacy_menu_supports_reference_and_native_choices(self):
        for answers,expected in [(['2','n'],['--stage2_attention','default']),
                                 (['','y'],['--stage2_attention','cudnn','--stage2_native_convolutions'])]:
            output=io.StringIO()
            with patch('builtins.input',side_effect=answers),redirect_stdout(output):
                args=run_interactive.prompt_gpu_speed_args()
            self.assertEqual(args,expected)
            self.assertIn('quality may degrade',output.getvalue().lower())

    def test_guided_speed_default_warns_and_selects_cudnn(self):
        output=io.StringIO()
        with patch('builtins.input',side_effect=['']*40),redirect_stdout(output):
            settings=launcher.prompt_settings(has_sequences=False)
        self.assertEqual(getattr(settings,'stage2_attention',None),'cudnn')
        self.assertIn('quality may degrade',output.getvalue().lower())
        self.assertEqual(self.config_for(ROOT/'source.mp4',settings)['cfg'].stage2_attention,'cudnn')

    def test_explicit_speed_options_reach_runtime_and_reference_remains_available(self):
        settings=launcher.Settings()
        settings.stage2_attention='default'
        settings.stage2_native_convolutions=True
        settings.preprocess_device='cuda'
        settings.cache_foundations=True
        cfg=self.config_for(ROOT/'source.mp4',settings)['cfg']
        self.assertEqual(cfg.stage2_attention,'default')
        self.assertTrue(getattr(cfg,'stage2_native_convolutions',False))
        self.assertEqual(cfg.preprocess_device,'cuda')
        self.assertTrue(cfg.cache_foundations)

    def test_real_menu_back_navigation_and_advanced_speed_selections(self):
        answers=['']*8+['','b','2','y','y','y','y','','','','']
        with patch('builtins.input',side_effect=answers),redirect_stdout(io.StringIO()):
            settings=launcher.prompt_settings(has_sequences=False)
        self.assertEqual(settings.stage2_attention,'default')
        self.assertEqual(settings.preprocess_device,'cuda')
        self.assertTrue(settings.cache_foundations)
        self.assertTrue(settings.stage2_native_convolutions)
        cfg=self.config_for(ROOT/'source.mp4',settings)['cfg']
        self.assertEqual(cfg.stage2_attention,'default')
        self.assertEqual(cfg.preprocess_device,'cuda')
        self.assertTrue(cfg.cache_foundations)
        self.assertTrue(cfg.stage2_native_convolutions)

    def config_for(self,source,settings,hint=None):
        command=launcher.inference_command(
            launcher.PreparedMedia(source,settings.sequence_fps),hint,ROOT/'out_selection_test',
            settings,ROOT/'ffmpeg.exe',ROOT/'tmp'/'selection_cache')
        captured={}
        model=object()
        def boundary(actual_model,input_path,output_dir,cfg,writer,**kwargs):
            captured.update(model=actual_model,input=input_path,cfg=cfg,writer=writer,**kwargs)
        with patch.object(runtime,'run_inference',side_effect=boundary),redirect_stdout(io.StringIO()):
            returned=infer.main(command[2:],model=model)
        self.assertIs(returned,model)
        self.assertIs(captured['model'],model)
        return captured

    def test_guided_video_color_range_hints_and_exports_reach_runtime(self):
        for i,(color,levels) in enumerate((c,r) for c in ('srgb','rec709','linear')
                                         for r in ('auto','limited','full')):
            for output_format in ('video','exr'):
                hint_mode=('none','first','full')[i%3]
                settings=launcher.Settings(input_color=color,input_range=levels,
                    hint_mode=hint_mode,hint_quality=.8,authoritative=(hint_mode=='full'),
                    output_format=output_format,exr_dtype=('half','float')[i%2],
                    outputs=('alpha','cutout') if i%2 else ('fg','checker'),
                    cutout_linear=bool(i%2),despill_strength=.4,low_vram=bool(i%2),
                    hann_chunk=8,hann_stride=4,num_frames=13)
                with self.subTest(color=color,levels=levels,format=output_format):
                    hint=ROOT/'supplied_hint.png' if hint_mode!='none' else None
                    captured=self.config_for(ROOT/'source.mp4',settings,hint)
                    cfg=captured['cfg']
                    self.assertEqual((cfg.video_decoder,cfg.input_color,cfg.input_range),('ffmpeg',color,levels))
                    self.assertEqual((cfg.output_format,cfg.exr_dtype,cfg.outputs),
                                     (output_format,settings.exr_dtype,set(settings.outputs)))
                    self.assertEqual((cfg.cutout_linear,cfg.despill_strength),(settings.cutout_linear,.4))
                    self.assertEqual((cfg.hann_chunk,cfg.hann_stride,cfg.num_frames),(8,4,13))
                    self.assertEqual(cfg.low_vram,settings.low_vram)
                    self.assertFalse(cfg.auto_vram)
                    self.assertEqual(cfg.carry_hint,hint_mode!='full')
                    self.assertEqual(captured['initial_hint_path'],str(hint) if hint_mode=='first' else None)
                    self.assertEqual(captured['hint_video_path'],str(hint) if hint_mode=='full' else None)
                    if hint_mode!='none':
                        self.assertEqual(cfg.hint_quality,.8)

    def test_guided_sequence_fps_linear_interpretation_and_auto_memory_reach_runtime(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            settings=launcher.Settings(sequence_fps=23.976,outputs=('alpha',),output_format='exr',exr_dtype='float')
            cfg=self.config_for(Path(folder),settings)['cfg']
            self.assertTrue(cfg.frame_dir_linear)
            self.assertEqual(cfg.frame_dir_fps,23.976)
            self.assertTrue(cfg.auto_vram)
            self.assertEqual(cfg.outputs,{'alpha'})
            self.assertEqual((cfg.output_format,cfg.exr_dtype),('exr','float'))
