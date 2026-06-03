from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.rl_traffic.config import SumoConfig
from src.rl_traffic import sumo_adapter


class FakeSumoLib:
    @staticmethod
    def checkBinary(name: str) -> str:
        return name


class FakeTraci:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.commands: list[list[str]] = []
        self.closed = False
        self.output_path: Path | None = None

    def start(self, command: list[str]) -> None:
        self.commands.append(command)
        prefix = command[command.index("--output-prefix") + 1]
        self.output_path = self.output_root / "detector" / f"{prefix}detector.xml"
        self.output_path.write_text("temporary output", encoding="utf-8")

    def close(self) -> None:
        self.closed = True


class SumoSessionTests(unittest.TestCase):
    def test_session_uses_unique_output_prefix_and_cleans_prefixed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector_dir = root / "detector"
            detector_dir.mkdir()
            sumocfg = root / "grid.sumocfg"
            sumocfg.write_text("<configuration/>", encoding="utf-8")
            original_output = detector_dir / "detector.xml"
            original_output.write_text("original output", encoding="utf-8")
            fake_traci = FakeTraci(root)
            config = SumoConfig(sumocfg=str(sumocfg))

            with (
                patch.object(sumo_adapter, "sumolib", FakeSumoLib()),
                patch.object(sumo_adapter, "traci", fake_traci),
            ):
                session = sumo_adapter.SumoSession(config)
                session.start(seed=17)
                temporary_output = fake_traci.output_path
                self.assertIsNotNone(temporary_output)
                self.assertTrue(temporary_output.exists())
                session.close()

            self.assertTrue(fake_traci.closed)
            self.assertFalse(temporary_output.exists())
            self.assertEqual(original_output.read_text(encoding="utf-8"), "original output")


if __name__ == "__main__":
    unittest.main()
