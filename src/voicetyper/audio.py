from typing import Optional, Dict, Union

import pyaudio as _p


class AudioDeviceResolver:
    """
    封装 PyAudio 的输入设备枚举工具。

    设计意图：
    - 统一、稳定地获取默认输入设备与全部输入设备列表
    - 提供面向“用户可见端点”的过滤方法，尽量贴近系统声音面板
    - 支持上下文协议，确保 PortAudio 资源在退出时被正确释放

    注意事项：
    - 在 Windows 上，PortAudio 会枚举多个主机 API（如 MME、DirectSound、WASAPI 等）下的设备，
      这会导致列表项比系统“声音”面板更多（含回环/虚拟设备、重复名称）。
    - 对于只做设备查询的场景，推荐使用 with 管理生命周期，或显式调用 close() 释放资源。
    """
    def __init__(self) -> None:
        """
        初始化解析器并尝试创建 PyAudio 实例。

        若 PyAudio 初始化失败，_available 为 False，后续查询方法将返回 None 或空列表。
        """
        self._pa: Optional[_p.PyAudio] = None
        self._available: bool = False
        try:
            self._pa = _p.PyAudio()
            self._available = True
        except Exception:
            self._available = False

    def __enter__(self) -> "AudioDeviceResolver":
        """
        进入上下文管理，返回自身。

        返回：
            AudioDeviceResolver: 当前解析器实例。
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        退出上下文时自动释放底层 PortAudio 资源。

        参数：
            exc_type: 异常类型（若无异常为 None）
            exc: 异常对象（若无异常为 None）
            tb:   异常追踪（若无异常为 None）
        """
        self.close()

    def default_input(self) -> Optional[Dict[str, Union[int, str]]]:
        """
        获取系统默认输入设备的信息。

        返回：
            Optional[Dict[str, Union[int, str]]]: 
                - 成功：{"index": int, "name": str}
                - 失败或不可用：None
        """
        if not self._available or self._pa is None:
            return None
        try:
            dev = self._pa.get_default_input_device_info()
            if 'index' not in dev:
                return None
            return {"index": int(dev["index"]), "name": str(dev.get("name", "Unknown"))}
        except Exception:
            return None

    def list_inputs(self) -> list[Dict[str, Union[int, str]]]:
        """
        列出所有具有输入通道的设备（跨主机 API，原始视图）。

        返回：
            list[Dict[str, Union[int, str]]]: 每项形如 {"index": int, "name": str}
        """
        if not self._available or self._pa is None:
            return []
        out: list[Dict[str, Union[int, str]]] = []
        try:
            count = self._pa.get_device_count()
            for i in range(count):
                try:
                    info = self._pa.get_device_info_by_index(i)
                    if int(info.get("maxInputChannels", 0)) > 0:
                        out.append({
                            "index": int(info.get("index", i)),
                            "name": str(info.get("name", "Unknown")),
                        })
                except Exception:
                    continue
        except Exception:
            return out
        return out

    def list_user_endpoints(
        self,
        prefer_hostapi: Optional[str] = "Windows WASAPI",
        deduplicate: bool = True,
        exclude_keywords: Optional[list[str]] = None
    ) -> list[Dict[str, Union[int, str]]]:
        """
        列出更贴近“用户面板”的输入端点列表。

        参数：
            prefer_hostapi: 优先保留的主机 API 名称（Windows 默认 "Windows WASAPI"）。设为 None 则不过滤主机 API。
            deduplicate: 是否按名称去重（忽略大小写与首尾空白）。
            exclude_keywords: 需要排除的名称关键词（不区分大小写），默认 ["loopback"]。

        返回：
            list[Dict[str, Union[int, str]]]: 每项 {"index": int, "name": str}
        """
        if not self._available or self._pa is None:
            return []
        if exclude_keywords is None:
            exclude_keywords = ["loopback"]
        result: list[Dict[str, Union[int, str]]] = []
        try:
            count = self._pa.get_device_count()
            for i in range(count):
                try:
                    info = self._pa.get_device_info_by_index(i)
                    if int(info.get("maxInputChannels", 0)) <= 0:
                        continue
                    hostapi_idx = int(info.get("hostApi", -1))
                    hostapi_name = ""
                    try:
                        hai = self._pa.get_host_api_info_by_index(hostapi_idx)
                        hostapi_name = str(hai.get("name", ""))
                    except Exception:
                        hostapi_name = ""
                    if prefer_hostapi and hostapi_name != prefer_hostapi:
                        continue
                    name = str(info.get("name", "Unknown"))
                    lowered = name.lower()
                    if any(k in lowered for k in exclude_keywords):
                        continue
                    result.append({"index": int(info.get("index", i)), "name": name})
                except Exception:
                    continue
        except Exception:
            return result
        if deduplicate:
            seen: set[str] = set()
            unique: list[Dict[str, Union[int, str]]] = []
            for item in result:
                key = str(item["name"]).strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                unique.append(item)
            return unique
        return result

    def close(self) -> None:
        """
        显式释放 PyAudio/PortAudio 资源。

        可安全重复调用；在 with 块中由 __exit__ 自动调用。
        """
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
