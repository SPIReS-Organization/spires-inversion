import hashlib
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "spires_inversion"


def test_compiled_extension_matches_cpp_source():
    stamp = PKG / "_core.src.sha256"
    if not (PKG / "spires.cpp").exists():
        pytest.skip("installed wheel: no C++ sources alongside the extension")
    if not stamp.exists():
        pytest.fail("no _core.src.sha256 next to the extension; rebuild with "
                    "`python setup.py build_ext --inplace`")
    h = hashlib.sha256()
    for name in ("spires.cpp", "spires.h"):
        h.update((PKG / name).read_bytes())
    assert stamp.read_text().strip() == h.hexdigest(), (
        "spires.cpp/spires.h changed since _core was built; rebuild with "
        "`python setup.py build_ext --inplace`")
