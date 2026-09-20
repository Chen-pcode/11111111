import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from author_protocol import ABLATIONS, make_config
from models.egeunet import EGEUNet
from research_support import paired_files
from utils import GT_BceDiceLoss


ROOT = Path(__file__).resolve().parents[1]


class PairingTests(unittest.TestCase):
    def test_isic_ids_and_missing_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, masks = root / 'train/images', root / 'train/masks'
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            (images / 'ISIC_001.jpg').touch()
            (masks / 'ISIC_001_segmentation.png').touch()
            (images / 'ISIC_002.png').touch()
            (masks / 'ISIC_002.png').touch()
            (images / 'notes.txt').touch()
            self.assertEqual([row[0] for row in paired_files(root, 'train')], ['ISIC_001', 'ISIC_002'])
            (masks / 'ISIC_002.png').rename(masks / 'ISIC_003.png')
            with self.assertRaisesRegex(ValueError, 'Unpaired files'):
                paired_files(root, 'train')

    def test_missing_directory_reports_data_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, '--data'):
                paired_files(directory, 'train')


class FrequencyTests(unittest.TestCase):
    def test_all_presets_have_finite_gradients_and_shared_initial_weights(self):
        torch.set_num_threads(1)
        baseline = None
        for experiment in ABLATIONS:
            with self.subTest(experiment=experiment):
                config = make_config(['--experiment', experiment, '--device', 'cpu'])
                model = EGEUNet(**config.model_config)
                if baseline is None:
                    baseline = {name: value.clone() for name, value in model.state_dict().items()}
                else:
                    for name, value in baseline.items():
                        self.assertTrue(torch.equal(model.state_dict()[name], value), name)
                stages = [stage for stage in range(1, 6) if hasattr(getattr(model, f'GAB{stage}'), 'frequency')]
                self.assertEqual(stages, [] if experiment == 'A0' else [1, 2, 3])
                auxiliary, prediction = model(torch.randn(2, 3, 32, 32))
                self.assertEqual(prediction.shape, (2, 1, 32, 32))
                self.assertTrue(all(item.shape == prediction.shape for item in auxiliary))
                loss = GT_BceDiceLoss()(auxiliary, prediction, torch.rand_like(prediction))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in model.named_parameters():
                    if '.frequency.' in name:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class TrainingTests(unittest.TestCase):
    def run_cli(self, *arguments, success=True):
        completed = subprocess.run(
            [sys.executable, 'train.py', *arguments], cwd=ROOT,
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120,
        )
        output = completed.stdout + completed.stderr
        if success:
            self.assertEqual(completed.returncode, 0, output)
        else:
            self.assertNotEqual(completed.returncode, 0, output)
        return output

    def test_train_evaluate_resume_and_output_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            rng = np.random.default_rng(42)
            for split in ('train', 'val'):
                for kind in ('images', 'masks'):
                    (data / split / kind).mkdir(parents=True)
                for index in range(2):
                    sample_id = f'{split}_{index}'
                    image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                    mask = np.zeros((32, 32), dtype=np.uint8)
                    mask[8:24, 8:24] = 255
                    Image.fromarray(image).save(data / split / 'images' / f'{sample_id}.png')
                    Image.fromarray(mask).save(data / split / 'masks' / f'{sample_id}_segmentation.png')
            for experiment in ('A0', 'A6'):
                with self.subTest(experiment=experiment):
                    run = root / experiment
                    args = ['--data', str(data), '--experiment', experiment, '--device', 'cpu',
                            '--epochs', '1', '--batch-size', '1', '--size', '32', '--out', str(run)]
                    output = self.run_cli(*args)
                    self.assertIn(f'Experiment={experiment}', output)
                    self.assertIn('active_stages=[]' if experiment == 'A0' else 'active_stages=[1, 2, 3]', output)
                    spec = json.loads((run / 'config.json').read_text(encoding='utf-8'))
                    self.assertEqual(spec['settings']['experiment'], experiment)
                    self.assertFalse(spec['independent_test'])
                    self.assertTrue((run / 'manifest.json').is_file())
                    checkpoint = torch.load(run / 'checkpoints/latest.pth', map_location='cpu', weights_only=True)
                    self.assertEqual(checkpoint['epoch'], 1)
                    self.assertEqual(checkpoint['step'], 2)
                    frequency_keys = [key for key in checkpoint['model_state_dict'] if '.frequency.' in key]
                    self.assertEqual(bool(frequency_keys), experiment == 'A6')
                    self.assertTrue((run / 'checkpoints/best.pth').is_file())
                    self.run_cli('--out', str(run), '--evaluate', '--device', 'cpu')
                    self.run_cli('--out', str(run), '--resume', '--device', 'cpu')
                    self.assertIn('Run directory is not empty', self.run_cli(*args, success=False))


if __name__ == '__main__':
    unittest.main()
