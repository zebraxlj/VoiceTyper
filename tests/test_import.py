import voicetyper


def test_imports():
    assert hasattr(voicetyper, 'AudioDeviceResolver')
    assert hasattr(voicetyper, 'BackgroundSTT')
    assert hasattr(voicetyper, 'AsrEngine')
    assert hasattr(voicetyper, 'PushToTalkRecorder')
    assert hasattr(voicetyper, 'RecorderConfig')
    assert hasattr(voicetyper, 'ResourceMonitor')
    print("Imports successful!")


if __name__ == "__main__":
    test_imports()
