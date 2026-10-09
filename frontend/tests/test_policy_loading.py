"""An isolated checkout must reuse the verified root rule and reject drift."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT.parent / "source/autodub_v2_scheme3_lighttts_final/scheme3_policy.py"
if not POLICY.is_file():
    POLICY = ROOT.parent / "scheme3_policy.py"


class PolicyLoadingTest(unittest.TestCase):
    def check_checkout(self, policy_bytes):
        (ROOT / '.tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as tmp:
            repo = Path(tmp)
            ui = repo / 'frontend'
            ui.mkdir()
            for name in ['mock_backend.py', 'backend_client.py', 'fixtures.py', 'routing_explanation.py']:
                shutil.copyfile(ROOT / name, ui / name)
            if policy_bytes is not None:
                (repo / 'scheme3_policy.py').write_bytes(policy_bytes)
            result = subprocess.run([sys.executable, '-B', '-c', 'import mock_backend'],
                                    cwd=ui, capture_output=True, timeout=20)
            self.assertFalse((repo / '__pycache__').exists())
            return result

    def test_verified_root_policy_loads_without_stage_source(self):
        self.assertEqual(self.check_checkout(POLICY.read_bytes()).returncode, 0)

    def test_different_rule_version_is_rejected(self):
        result = self.check_checkout(POLICY.read_bytes() + b'\n# changed\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'RuntimeError', result.stderr)

    def test_windows_git_crlf_checkout_is_accepted(self):
        data = POLICY.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        self.assertEqual(self.check_checkout(data).returncode, 0)

    def test_missing_policy_is_reported(self):
        result = self.check_checkout(None)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'FileNotFoundError', result.stderr)
