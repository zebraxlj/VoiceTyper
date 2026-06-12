"""Audio input-device hot-plug watcher.

This is the **only** platform-specific module in the reusable layer. On Windows
it installs a WASAPI ``IMMNotificationClient`` and fires a callback when audio
endpoints change; on other platforms (or when ``comtypes`` is unavailable)
:meth:`DeviceChangeWatcher.start` returns ``False`` and the caller should fall
back to re-enumerating on demand (e.g. when the picker is opened).

Kept separate from :mod:`voicetyper.devices` so that module stays 100%
cross-platform and dependency-light: non-Windows consumers never import the COM
machinery here.
"""

import logging
import sys
import threading
from typing import Optional

logger = logging.getLogger("voicetyper.device_watch")

_COM_TYPES: Optional[dict] = None


def _is_render_endpoint(device_id) -> bool:
    """判断一个 WASAPI endpoint ID 是否为输出(render)设备。

    MMDevice endpoint ID 形如 ``{0.0.<flow>.00000000}.{guid}``，第三段 0=render
    (输出)、1=capture(输入)。我们只关心输入设备，故用它过滤掉输出设备的通知。
    保守判断：只在“确定是 render”时返回 True；未知格式按非 render 处理（照常触发
    刷新），避免万一 ID 格式变化时漏掉输入设备的变更。
    """
    return bool(device_id) and str(device_id).startswith("{0.0.0.")


def _load_com_types() -> dict:
    """惰性定义 WASAPI 端点通知所需的 COM 接口，返回相关类型。

    需要 ``comtypes``；不可用时抛异常。首次调用后缓存，接口类只定义一次
    （重复定义会触发 comtypes 的 GUID 冲突告警）。
    """
    global _COM_TYPES
    if _COM_TYPES is not None:
        return _COM_TYPES

    from ctypes import HRESULT, POINTER, Structure
    from ctypes.wintypes import DWORD, LPCWSTR

    import comtypes
    from comtypes import COMObject, GUID, IUnknown, STDMETHOD

    class PROPERTYKEY(Structure):
        _fields_ = [("fmtid", GUID), ("pid", DWORD)]

    class IMMNotificationClient(IUnknown):
        _iid_ = GUID("{7991EEC9-7E89-4D85-8390-6C703CEC60C0}")
        _methods_ = [
            STDMETHOD(HRESULT, "OnDeviceStateChanged", [LPCWSTR, DWORD]),
            STDMETHOD(HRESULT, "OnDeviceAdded", [LPCWSTR]),
            STDMETHOD(HRESULT, "OnDeviceRemoved", [LPCWSTR]),
            STDMETHOD(HRESULT, "OnDefaultDeviceChanged", [DWORD, DWORD, LPCWSTR]),
            STDMETHOD(HRESULT, "OnPropertyValueChanged", [LPCWSTR, PROPERTYKEY]),
        ]

    class IMMDeviceEnumerator(IUnknown):
        _iid_ = GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
        _methods_ = [
            STDMETHOD(HRESULT, "EnumAudioEndpoints",
                      [DWORD, DWORD, POINTER(POINTER(IUnknown))]),
            STDMETHOD(HRESULT, "GetDefaultAudioEndpoint",
                      [DWORD, DWORD, POINTER(POINTER(IUnknown))]),
            STDMETHOD(HRESULT, "GetDevice",
                      [LPCWSTR, POINTER(POINTER(IUnknown))]),
            STDMETHOD(HRESULT, "RegisterEndpointNotificationCallback",
                      [POINTER(IMMNotificationClient)]),
            STDMETHOD(HRESULT, "UnregisterEndpointNotificationCallback",
                      [POINTER(IMMNotificationClient)]),
        ]

    class _NotificationClient(COMObject):
        _com_interfaces_ = [IMMNotificationClient]

        def __init__(self, on_event) -> None:
            super().__init__()
            self._on_event = on_event

        # 注意：comtypes 走 call_with_this，会把 COM 的 this 指针作为首个参数传入，
        # 故每个方法都需显式接收 `this`。
        # 只响应输入(capture)设备的变更；输出(render)设备的通知与输入列表无关，
        # 全部忽略（否则复合设备插拔时会被输出端的通知额外触发一次无意义的重扫）。
        def OnDeviceStateChanged(self, this, pwstrDeviceId, dwNewState):
            logger.debug("IMMNotification: OnDeviceStateChanged state=%s id=%s",
                         dwNewState, pwstrDeviceId)
            if not _is_render_endpoint(pwstrDeviceId):
                self._on_event()
            return 0

        def OnDeviceAdded(self, this, pwstrDeviceId):
            logger.debug("IMMNotification: OnDeviceAdded id=%s", pwstrDeviceId)
            if not _is_render_endpoint(pwstrDeviceId):
                self._on_event()
            return 0

        def OnDeviceRemoved(self, this, pwstrDeviceId):
            logger.debug("IMMNotification: OnDeviceRemoved id=%s", pwstrDeviceId)
            if not _is_render_endpoint(pwstrDeviceId):
                self._on_event()
            return 0

        def OnDefaultDeviceChanged(self, this, flow, role, pwstrDefaultDeviceId):
            logger.debug("IMMNotification: OnDefaultDeviceChanged flow=%s role=%s id=%s",
                         flow, role, pwstrDefaultDeviceId)
            if flow != 0:  # 0 = eRender(输出)；只响应输入/全部
                self._on_event()
            return 0

        def OnPropertyValueChanged(self, this, pwstrDeviceId, key):
            logger.debug("IMMNotification: OnPropertyValueChanged id=%s", pwstrDeviceId)
            return 0

    _COM_TYPES = {
        "comtypes": comtypes,
        "CLSID_MMDeviceEnumerator": GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}"),
        "IMMDeviceEnumerator": IMMDeviceEnumerator,
        "NotificationClient": _NotificationClient,
    }
    return _COM_TYPES


class DeviceChangeWatcher:
    """音频设备热插拔变化时回调通知。

    仅 Windows 实现（WASAPI ``IMMNotificationClient``）。其它平台或 ``comtypes``
    不可用时，:meth:`start` 返回 ``False``，调用方应回退到"按需重新枚举"
    （例如打开下拉时）。

    回调在内部线程触发，消费方需自行把 UI 操作切回各自的 UI 线程；OS 短时间内
    连发的多个事件会被合并（防抖）。
    """

    def __init__(self, debounce_s: float = 0.4) -> None:
        self._debounce_s = debounce_s
        self._on_change = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._ready_evt = threading.Event()
        self._started = False
        self._timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()

    @property
    def supported(self) -> bool:
        return sys.platform == "win32"

    def start(self, on_change) -> bool:
        """开始监听。成功安装 OS 监听返回 True，否则 False（调用方应轮询）。"""
        if not self.supported or self._thread is not None:
            return False
        try:
            # 在调用线程（主线程）预导入 comtypes：它在导入期会 CoInitialize(STA)，
            # 必须发生在这里，否则会与工作线程的 MTA 初始化冲突。
            _load_com_types()
        except Exception:
            logger.exception("comtypes 不可用，设备监听回退到轮询")
            return False
        self._on_change = on_change
        self._stop_evt.clear()
        self._ready_evt.clear()
        self._started = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready_evt.wait(timeout=3.0)
        if not self._started:
            self._thread = None
        return self._started

    def stop(self) -> None:
        """停止监听并释放资源。可安全重复调用。"""
        self._stop_evt.set()
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._thread = None

    def _debounced_fire(self) -> None:
        # 来自 COM 线程，可能短时间连发多次；合并为一次。
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_s, self._emit)
            self._timer.daemon = True
            self._timer.start()

    def _emit(self) -> None:
        if self._stop_evt.is_set():
            return
        cb = self._on_change
        if cb is not None:
            logger.debug("设备变更防抖触发，通知刷新")
            try:
                cb()
            except Exception:
                logger.exception("设备变更回调执行失败")

    def _run(self) -> None:
        import ctypes

        enumerator = None
        client = None
        co_init = False
        try:
            # MTA：COM 回调在 RPC 线程直接投递，无需消息循环。
            ctypes.windll.ole32.CoInitializeEx(None, 0x0)
            co_init = True
            types = _load_com_types()
            comtypes = types["comtypes"]
            enumerator = comtypes.CoCreateInstance(
                types["CLSID_MMDeviceEnumerator"],
                interface=types["IMMDeviceEnumerator"],
                clsctx=0x17,  # CLSCTX_ALL
            )
            client = types["NotificationClient"](self._debounced_fire)
            enumerator.RegisterEndpointNotificationCallback(client)
            self._started = True
            self._ready_evt.set()
            self._stop_evt.wait()  # 保持 COM 套间存活
        except Exception:
            logger.exception("音频设备变更监听启动失败，将回退到轮询")
            self._ready_evt.set()
        finally:
            try:
                if enumerator is not None and client is not None:
                    enumerator.UnregisterEndpointNotificationCallback(client)
            except Exception:
                pass
            if co_init:
                try:
                    ctypes.windll.ole32.CoUninitialize()
                except Exception:
                    pass
