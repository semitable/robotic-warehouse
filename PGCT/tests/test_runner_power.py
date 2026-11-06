"""pgct_runner.power: opting a training process out of Windows background power throttling (E98)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # PGCT/ (pgct_runner)

from pgct_runner.power import disable_power_throttling  # noqa: E402


def test_opt_out_succeeds_on_windows_and_is_a_no_op_elsewhere():
    result = disable_power_throttling()
    assert result is (sys.platform == "win32")
    assert disable_power_throttling() is result  # idempotent
