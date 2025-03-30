#!/usr/bin/env python3
# test_Linux_Install.sh
# This script is used to test the Linux install/Update & Run scripts.
#
# Usage: python -m unittest test_Linux_Install.sh
#
# Imports
import unittest
import subprocess
import os
import shutil
import tempfile
import signal
import filecmp
#
####################################################################################################
#
# Functions:


class TestTLDWScripts(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.install_script = os.path.join(self.test_dir, "Linux_Install_Update.sh")
        self.run_script = os.path.join(self.test_dir, "Linux_Run_tldw.sh")
        self.original_run_script = os.path.join(self.test_dir, "original_Linux_Run_tldw.sh")
        self.processes_to_kill = []

        try:
            shutil.copy("/path/to/Linux_Install_Update.sh", self.install_script)
            shutil.copy("/path/to/Linux_Run_tldw.sh", self.run_script)
            shutil.copy("/path/to/Linux_Run_tldw.sh", self.original_run_script)
        except IOError as e:
            self.fail(f"Unable to copy scripts: {e}")

        for script in [self.install_script, self.run_script, self.original_run_script]:
            os.chmod(script, 0o755)

    def tearDown(self):
        for process in self.processes_to_kill:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass  # Process has already terminated
        shutil.rmtree(self.test_dir)

    def execute_script(self, script, input_text=None, timeout=60):
        try:
            process = subprocess.Popen(
                ['/bin/bash', script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.test_dir,
                text=True,
                preexec_fn=os.setsid
            )
            self.processes_to_kill.append(process)
            stdout, stderr = process.communicate(input=input_text, timeout=timeout)
            return process.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            return -1, "", "Timeout"

    def modify_run_script(self, test_mode=True):
        with open(self.run_script, 'r') as f:
            content = f.read()
        if test_mode:
            content = content.replace('python3 summarize.py -gui', 'python3 summarize.py -gui --test-mode')
        else:
            content = content.replace('python3 summarize.py -gui --test-mode', 'python3 summarize.py -gui')
        with open(self.run_script, 'w') as f:
            f.write(content)

    def test_fresh_install(self):
        returncode, stdout, stderr = self.execute_script(self.install_script, "n\nn\n")
        self.assertEqual(returncode, 0, f"Fresh install failed: {stderr}")
        self.assertIn("Installation/Update completed successfully!", stdout)

    def test_update(self):
        self.execute_script(self.install_script, "n\nn\n")
        returncode, stdout, stderr = self.execute_script(self.install_script, "y\ny\n")
        self.assertEqual(returncode, 0, f"Update failed: {stderr}")
        self.assertIn("Installation/Update completed successfully!", stdout)

    def test_gpu_install(self):
        returncode, stdout, stderr = self.execute_script(self.install_script, "n\ny\n1\n")
        self.assertEqual(returncode, 0, f"GPU install failed: {stderr}")
        self.assertIn("Installation/Update completed successfully!", stdout)

        gpu_choice_file = os.path.join(self.test_dir, "tldw", "gpu_choice.txt")
        self.assertTrue(os.path.exists(gpu_choice_file), "GPU choice file not created")
        with open(gpu_choice_file, "r") as f:
            self.assertEqual(f.read().strip(), "cuda")

    def test_run_script(self):
        self.execute_script(self.install_script, "n\nn\n")
        self.modify_run_script(test_mode=True)

        returncode, stdout, stderr = self.execute_script(self.run_script, timeout=30)
        self.assertEqual(returncode, 0, f"Run script failed: {stderr}")
        self.assertIn("TLDW has been ran", stdout)

        self.modify_run_script(test_mode=False)
        self.assertTrue(filecmp.cmp(self.run_script, self.original_run_script), "Run script was not properly reverted")

    def test_run_script_long_running(self):
        self.execute_script(self.install_script, "n\nn\n")
        returncode, stdout, stderr = self.execute_script(self.run_script, timeout=10)
        self.assertEqual(returncode, -1, "Run script should have been terminated")
        self.assertIn("Timeout", stderr)


if __name__ == '__main__':
    unittest.main()

# End of File
####################################################################################################
