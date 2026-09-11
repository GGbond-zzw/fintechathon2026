from pathlib import Path
from unittest import TestCase

from src.utils.paths import ProjectPaths


class ProjectPathsTest(TestCase):
    def test_paths_are_sourced_from_config(self) -> None:
        names = list(ProjectPaths.__annotations__)
        config = {"paths": {name: f"D:/project/{name}" for name in names}}
        paths = ProjectPaths.from_config(config)
        self.assertEqual(paths.root, Path("D:/project/root"))
        self.assertEqual(paths.test, Path("D:/project/test"))

