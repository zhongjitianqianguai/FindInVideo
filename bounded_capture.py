"""为可能失联的视频路径提供有界 OpenCV 读取。"""

import math
import os
import pickle
import queue
import subprocess
import sys
import threading
import traceback
from typing import Any


_PIPE_CLOSED = object()


class _ReaderFailure:
    def __init__(self, message: str):
        self.message = message


class BoundedVideoCapture:
    """在独立子进程中读取视频，避免共享盘 I/O 无限阻塞主进程。"""

    def __init__(self, path: os.PathLike[str] | str, timeout: float = 30.0):
        self.path = os.fsdecode(os.fspath(path))
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('timeout 必须是大于零的有限秒数')

        self._request_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._responses: queue.Queue[Any] = queue.Queue()
        self._release_requested = False
        self._closed = False
        self._terminated = False
        self._opened = False
        self._process = None
        self._reader = None

        creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0
        self._process = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), '--worker', self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            creationflags=creationflags,
        )
        self._reader = threading.Thread(
            target=self._reader_pump,
            name='BoundedVideoCapture-reader',
            daemon=True,
        )
        try:
            self._reader.start()
            response = self._wait_response(self.timeout, '打开视频')
            self._opened = bool(response.get('opened', False))
        except BaseException:
            self._terminate_worker()
            raise

    def isOpened(self) -> bool:
        """返回视频是否已成功打开且尚未释放。"""
        with self._state_lock:
            return self._opened and not self._closed

    def get(self, prop: int) -> float:
        """读取视频属性。"""
        if not self.isOpened():
            return 0.0
        response = self._request('get', prop=prop)
        return float(response['value'])

    def set(self, prop: int, value: float) -> bool:
        """设置视频属性。"""
        if not self.isOpened():
            return False
        response = self._request('set', prop=prop, value=value)
        return bool(response['value'])

    def read(self) -> tuple[bool, Any]:
        """读取下一帧；文件结束时返回 ``(False, None)``。"""
        if not self.isOpened():
            return False, None
        response = self._request('read')
        success, frame = response['value']
        return bool(success), frame if success else None

    def release(self) -> None:
        """释放视频；若读取正在阻塞，则先终止其独占子进程。"""
        with self._state_lock:
            if self._closed or self._release_requested:
                return
            self._release_requested = True

        acquired = self._request_lock.acquire(timeout=0.05)
        if not acquired:
            self._terminate_worker()
            acquired = self._request_lock.acquire(timeout=2.0)
            if not acquired:
                return

        try:
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    self._send_request_locked('release', timeout=min(self.timeout, 1.0))
                except (OSError, TimeoutError):
                    # 释放属于清理操作，不能覆盖调用方正在处理的异常。
                    pass
            self._terminate_worker()
        finally:
            self._request_lock.release()

    def _request(self, command: str, **fields: Any) -> dict[str, Any]:
        with self._request_lock:
            with self._state_lock:
                if self._closed or self._release_requested:
                    raise OSError('视频读取进程已关闭')
            return self._send_request_locked(command, **fields)

    def _send_request_locked(
        self,
        command: str,
        timeout: float | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            raise OSError('视频读取进程不可用')

        try:
            pickle.dump({'command': command, **fields}, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError, EOFError) as exc:
            self._terminate_worker()
            raise OSError(f'视频读取进程通信失败: {exc}') from exc

        try:
            response = self._wait_response(self.timeout if timeout is None else timeout, command)
        except TimeoutError:
            self._terminate_worker()
            raise
        except OSError:
            self._terminate_worker()
            raise

        if not response.get('ok', False):
            raise OSError(response.get('error', '视频读取进程发生错误'))
        return response

    def _wait_response(self, timeout: float, operation: str) -> dict[str, Any]:
        try:
            item = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f'{operation}超过 {timeout:g} 秒仍未完成') from exc

        if item is _PIPE_CLOSED:
            code = self._process.poll() if self._process is not None else None
            raise OSError(f'视频读取进程的通信管道已关闭，退出码: {code}')
        if isinstance(item, _ReaderFailure):
            raise OSError(f'读取视频响应失败: {item.message}')
        if not isinstance(item, dict):
            raise OSError('视频读取进程返回了无效响应')
        if not item.get('ok', False):
            raise OSError(item.get('error', '视频读取进程发生错误'))
        return item

    def _reader_pump(self) -> None:
        process = self._process
        stream = process.stdout if process is not None else None
        if stream is None:
            self._responses.put(_ReaderFailure('标准输出管道不可用'))
            return
        try:
            while True:
                self._responses.put(pickle.load(stream))
        except EOFError:
            self._responses.put(_PIPE_CLOSED)
        except Exception as exc:
            self._responses.put(_ReaderFailure(str(exc)))

    def _terminate_worker(self) -> None:
        with self._cleanup_lock:
            with self._state_lock:
                self._closed = True
                self._release_requested = True
            if self._terminated:
                return
            self._terminated = True

            process = self._process
            if process is not None:
                try:
                    running = process.poll() is None
                except OSError:
                    running = False
                if running:
                    try:
                        process.terminate()
                    except OSError:
                        pass
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                except (OSError, ValueError):
                    pass
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
                    except OSError:
                        pass
                except OSError:
                    pass

                reader = self._reader
                if (
                    reader is not None
                    and reader is not threading.current_thread()
                    and reader.is_alive()
                ):
                    try:
                        reader.join(timeout=1.0)
                    except RuntimeError:
                        pass
                if reader is not None and reader.is_alive():
                    try:
                        if process.stdout is not None:
                            process.stdout.close()
                    except (OSError, ValueError):
                        pass
                    try:
                        reader.join(timeout=0.5)
                    except RuntimeError:
                        pass
                try:
                    if process.stdout is not None and not process.stdout.closed:
                        process.stdout.close()
                except (OSError, ValueError):
                    pass


def _worker_main(path: str) -> int:
    """运行只持有 OpenCV 对象的子进程协议循环。"""
    capture = None
    crash_stream = None

    def send(message: dict[str, Any]) -> None:
        pickle.dump(message, sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stdout.buffer.flush()

    try:
        import faulthandler
        crash_path = os.environ.get('FINDINVIDEO_CRASH_LOG_PATH') or os.path.join(os.path.dirname(__file__), 'logs', 'crash.log')
        try:
            os.makedirs(os.path.dirname(os.path.abspath(crash_path)), exist_ok=True)
            crash_stream = open(crash_path, 'a', encoding='utf-8')
            crash_stream.write(f'视频读取进程 pid={os.getpid()} parent_pid={os.getppid()} path={path}\n')
            crash_stream.flush()
            os.dup2(crash_stream.fileno(), 2)
            faulthandler.enable(file=crash_stream, all_threads=True)
        except OSError:
            # 主进程仍会记录真实退出状态，诊断路径不可写不应改变视频处理。
            pass
        import cv2
        import numpy as np  # noqa: F401

        capture = cv2.VideoCapture(path)
        send({'ok': True, 'opened': bool(capture.isOpened())})
        while True:
            try:
                request = pickle.load(sys.stdin.buffer)
            except EOFError:
                break

            command = request.get('command')
            try:
                if command == 'get':
                    value = capture.get(request['prop'])
                elif command == 'set':
                    value = capture.set(request['prop'], request['value'])
                elif command == 'read':
                    success, frame = capture.read()
                    value = (bool(success), frame if success else None)
                elif command == 'release':
                    capture.release()
                    send({'ok': True, 'value': None})
                    break
                else:
                    raise ValueError(f'未知视频操作: {command}')
                send({'ok': True, 'value': value})
            except Exception as exc:
                send({'ok': False, 'error': traceback.format_exc()})
    except BaseException as exc:
        try:
            send({'ok': False, 'error': traceback.format_exc()})
        except Exception:
            pass
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        if crash_stream is not None:
            faulthandler.disable()
            crash_stream.close()
    return 0


if __name__ == '__main__' and len(sys.argv) >= 3 and sys.argv[1] == '--worker':
    raise SystemExit(_worker_main(sys.argv[2]))
