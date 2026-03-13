import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import voicetyper


def test_imports():
    assert hasattr(voicetyper, 'AudioDeviceResolver')
    assert hasattr(voicetyper, 'BackgroundSTT')
    assert hasattr(voicetyper, 'AsrEngine')
    print("Imports successful!")


if __name__ == "__main__":
    test_imports()
