一个使用yolo11来检测视频中对象存在的时间位置的工具 
A tool that uses YOLO11 to detect the time and location of objects in videos

Face Model come from https://github.com/akanametov/yolo-face

### 存储空间不足时等待恢复

视频分段初始化及写入前会检查剩余空间（连续写入中每秒检查一次）。不足 64 MiB 时停止继续写入，保留已有检查点和已确认分段，进入恢复等待。空间恢复后从检查点续跑；合并时必须满足输入分段总大小加 64 MiB 的空间要求。

恢复等待默认每 5 分钟探测并输出一次状态，期间仍可主动暂停或按 Ctrl+C。原有 `FINDINVIDEO_RECOVERY_RETRY_SECONDS` 可覆盖默认间隔（单位秒，最小 1 秒）。新默认值也用于其他可恢复存储故障。已经启动的进程需要重新启动才会加载这次修改。
