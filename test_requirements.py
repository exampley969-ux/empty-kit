"""
Guards against a dependency being used in the code but missing from
requirements.txt. (Found the hard way: scipy, PyJWT and psutil were used by
Phases 3 and 9 but absent from the file, so a fresh install would crash on
import.)
"""
import ast
import os
import re
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# import name -> pip distribution name, where they differ
DIST = {"cv2": "opencv-python", "jwt": "PyJWT", "PIL": "Pillow", "yaml": "PyYAML"}


def third_party_imports():
    stdlib = set(sys.stdlib_module_names)
    found = set()
    for dirpath, _, files in os.walk(ROOT):
        if "tests" in os.path.relpath(dirpath, ROOT).split(os.sep):
            continue
        for name in files:
            if name.endswith(".py"):
                with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                    for node in ast.walk(ast.parse(f.read())):
                        if isinstance(node, ast.Import):
                            found |= {a.name.split(".")[0] for a in node.names}
                        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                            found.add(node.module.split(".")[0])
    return {m for m in found if m not in stdlib and m != "vision_radar"}


def declared_requirements():
    names = set()
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                names.add(re.split(r"[<>=!~\\[ ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


class TestRequirementsAreComplete(unittest.TestCase):
    def test_the_scan_is_not_vacuous(self):
        # A guard that silently scans nothing passes forever. (This exact
        # mistake was made in the first version of this test.)
        found = third_party_imports()
        for expected in ("cv2", "numpy", "scipy", "jwt", "psutil", "onnxruntime", "fastapi"):
            self.assertIn(expected, found)
        self.assertNotIn("unittest", found)                 # stdlib excluded
        self.assertGreaterEqual(len(declared_requirements()), 8)

    def test_a_missing_requirement_would_be_reported(self):
        declared = declared_requirements() - {"scipy"}
        self.assertIn("scipy", [m for m in third_party_imports()
                                if DIST.get(m, m).lower().replace("_", "-") not in declared])

    def test_every_third_party_import_is_declared(self):
        declared = declared_requirements()
        missing = []
        for module in sorted(third_party_imports()):
            dist = DIST.get(module, module).lower().replace("_", "-")
            if dist == "pydantic":          # installed as a dependency of fastapi
                dist = "fastapi"
            if dist not in declared:
                missing.append(f"{module} (pip: {dist})")
        self.assertEqual(missing, [], "used in code but not in requirements.txt")


if __name__ == "__main__":
    unittest.main()
