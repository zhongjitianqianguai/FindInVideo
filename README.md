一个使用yolo11来检测视频中对象存在的时间位置的工具 
A tool that uses YOLO11 to detect the time and location of objects in videos

Face Model come from https://github.com/akanametov/yolo-face

### 存储空间不足时等待恢复

视频分段初始化及写入前会检查剩余空间（连续写入中每秒检查一次）。不足 64 MiB 时停止继续写入，保留已有检查点和已确认分段，进入恢复等待。空间恢复后从检查点续跑；合并时必须满足输入分段总大小加 64 MiB 的空间要求。

恢复等待默认每 5 分钟探测并输出一次状态，期间仍可主动暂停或按 Ctrl+C。原有 `FINDINVIDEO_RECOVERY_RETRY_SECONDS` 可覆盖默认间隔（单位秒，最小 1 秒）。新默认值也用于其他可恢复存储故障。已经启动的进程需要重新启动才会加载这次修改。

`main.py` 启动时如果共享盘尚未挂载、输入暂时不可读，或共享数据库初始化遇到临时存储故障，也会按上述间隔等待并记录日志，恢复后继续原启动流程。检查针对实际输入路径；不会仅因父目录可访问就认定输入已恢复。可读取但确实为空的目录仍按原行为提示并退出，单视频输入仍受支持。

运行中，`main.py` 会在目录枚举、索引刷新、读取视频失败时核对存储根目录及实际文件访问情况。共享主机离线会触发同一恢复等待，不把目录当成已删除或已完成；恢复后重新扫描或从最近有效检查点重开视频，未保存的片段会重做。UNC 和映射共享盘的视频由独占辅助进程读取，打开、读帧和定位操作默认最多等待 30 秒，超时后回收该辅助进程，主终端继续处理恢复流程。存储可访问时，坏视频的解码失败和提前 EOF 保持原有失败、退避逻辑，不改为无限等待。原生读取进程的崩溃也写入 `crash.log`。

### 视频失败和程序退出诊断

四个视频入口会在导入模型和视频依赖之前启用本机诊断。日志只追加，不会在重启时覆盖：

- `logs/runtime_events.jsonl`：每次运行的 `run_id`、入口、PID、时间、当前视频，单视频失败的类型、消息、系统错误码和完整异常链，以及程序退出原因。先在本机 flush/fsync，再尽力镜像到 `directory_index.db` 的 `processing_events`；共享盘不可用时仍可查本机记录。
- `logs/crash.log`：原生访问冲突等致命错误发生时，faulthandler 尽力保存各线程 Python 堆栈，并记录运行及视频关联信息。
- `logs/launcher_events.log`：通过 `run_main*.bat` 视频启动脚本运行时，追加启动时间和 `main.py` 的真实退出码，即使 Python 未能写退出摘要也能留痕。
- `logs/launcher_startup_<token>.json`：保存随机 token、supervisor 与实际 `main.py` 进程的 PID、映像、创建时间、Windows 会话、控制台和 Job 校验结果；运行结束后追加真实退出码。

`process_exit.reason` 区分 `queue_exhausted`（本轮遍历结束，不保证全部视频成功）、`queue_exhausted_with_video_failures`（带单视频失败结束）、`pause_requested`、`keyboard_interrupt`、`system_exit` 和 `uncaught_exception`；摘要带失败次数及最后一次失败。主动关闭窗口、注销或关机的 Windows 控制通知记录为 `process_termination_requested`；恢复导致的入口重启会先记录 `process_restart_requested`。主动暂停源（SIGINT 或暂停文件）也会写入本机日志。

原有继续队列、失败退避、存储恢复、完成态及退出码语义保持不变。可用 `FINDINVIDEO_RUNTIME_LOG_PATH`、`FINDINVIDEO_CRASH_LOG_PATH` 覆盖两类诊断日志路径；无需新增配置，下次启动生效。没有真实异常的失败只记原因，不伪造堆栈。断电、强制杀进程等无法执行记录代码的情形，需结合最后运行事件、启动脚本退出码（若可获得）和 Windows 系统事件判断，不能保证生成退出摘要。

### 独立控制台启动

默认运行 `run_main.bat` 时，会由 `.venv\Scripts\python.exe` 调用 `independent_launcher.py`。启动器通过隐藏的 Windows PowerShell CIM broker 创建可见的新控制台 supervisor；supervisor 再以基础 Python 解释器启动固定的 `main.py`。随机 token 握手会核对 supervisor 和实际处理进程的 PID、映像、创建时间、Windows 会话、可见控制台及 Job 状态。任一检查失败或握手超时都会返回非零码，不会退回到附着启动。

BAT 返回 `0` 表示独立进程验收成功，视频处理在新控制台异步运行；`main.py` 的实际退出码随后写入 `logs/launcher_events.log` 和对应的启动结果 JSON。启动需要 Windows PowerShell CIM 与现有 `.venv`，不需要修改 `main.py` 或设置新环境变量。

存储恢复触发入口重启时，旧处理进程退出并释放模型、显存等资源，监护进程重新验证并启动处理进程，最终退出时再记录真实退出码。按 Ctrl+C 时由实际处理进程保存断点，监护进程等待其退出后再记录结果。
