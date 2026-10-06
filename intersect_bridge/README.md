# Portable Raman / Qt / motor / PCA-SVM 的 INTERSECT 桥接

这是独立新增项目。现有 `laser_test/main.cpp`、CMake、设备源码、原 EXE、模型和历史数据均不修改。
`ORNL/intersect-architecture` 是架构文档；这里实际使用 `intersect-sdk==0.9.3`。

## 已实现的范围

- INTERSECT 客户端查询状态、启动现有 `1d-x` / `1d-y` / `2d` 扫描、取消扫描、分页读取 SVM 结果。
- 从原 Qt 源码生成独立桥接版，原按钮和扫描逻辑继续工作。
- Qt 与 Python 用本机 JSON 文件邮箱通信。无需修改厂商驱动，也不需要给 Qt 增加网络库。
- Qt 的逐点采集、原子保存、motor ACK 保持原有顺序；SVM 在 Python 后台分析。
- 纯分类 PCA-SVM 加载已有 scaler、PCA、SVM，禁止在服务里训练。
- 请求 ID 去重、启动期限、Qt 启动实例 ID、执行前落盘记录。执行结果不明确时不自动重复动作。

第一版保留本地手动开启/关闭激光、采集暗谱和启用远程扫描。没有开放任意 XYZ 移动或远程激光功率控制。
`1d-x` / `1d-y` 仍是原程序的归零后 0–24 mm、间隔 2 mm、13 点扫描；`2d` 保持原有 Z/XY 路径。
客户端不能指定新的范围、步长。停止扫描仍不关闭激光。

## 文件布局

| 文件 | 用途 |
|---|---|
| `service.py` | INTERSECT 服务及请求模型 |
| `client.py` | 本机或另一台电脑运行的客户端 |
| `controller.py` | 绑定扫描会话、后台推理和结果分页 |
| `qt_client.py` | 本机文件邮箱客户端 |
| `svm_predictor.py` | 纯分类预处理及推理，无训练代码 |
| `prepare_qt.py` / `qt_bridge_members.inc` | 生成扩展后的 Qt 副本，检查替换锚点 |
| `build_qt.cmd` / `start_qt.cmd` | 编译/启动独立 Qt 版本 |
| `config.json` | 当前机器设置；包含账号密码，不应提交到仓库 |
| `config.example.json` | 配置模板 |
| `compose.yaml` / `enabled_plugins` | 本地 RabbitMQ MQTT5 消息代理配置 |
| `replay.py` | 历史逐点 JSON 回放，不启动任何设备 |
| `test_bridge.py` / `test_mock_integration.py` | 推理与无硬件模拟测试 |

生成文件放 `generated/`，编译产物放 `build_qt/bin/`；模拟版分别放 `generated_mock/`、`build_mock/bin/`。
新增扫描放 `build_qt/bin/scan_sessions/`；结果放 `runtime/results/<session_id>/`。
历史回放输出放 `runtime/replay.json`。这些都是本目录内的新文件，不会覆盖原扫描。

## 当前机器已经完成的准备

新增 `.venv` 已安装 SDK 和推理依赖。设备版与模拟版 Qt 使用本机 Qt 6.6.3 / MSVC2019 构建。
设备版运行时读取原 `handshake_v2/backend/SampleApl.exe`，并保持原 Downloads 下的电机 EXE 路径。
设备版使用与原 Qt 相同的 `controller.lock`，两者不能同时占用仪器。
生成 Qt 副本的原始源文件哈希记录在 `generated/source_manifest.json`。

本地 SDK 服务/schema 校验、真实历史 13 点回放、预处理一致性和模拟 Qt 联调可以在不连接硬件的情况下运行。
完整消息代理通信及真实硬件联调需在实际启动后验证；不能将模拟结果当作真机结果。

## 启动：先在同一台电脑验证

在 PowerShell 进入本目录：

```powershell
Set-Location 'C:\Users\Jeremy666\OneDrive\Documents\laser_test\intersect_bridge'
```

### 1. 配置模型与暗谱

编辑 `config.json`：

- `model_dir`：已保存的可信模型目录。第一版使用 `_PCA_SVM_r6g20260922_results` 的 pure 分支。
- `dark_file`：本次实验适用的完整暗谱。**模板预填的是 2026-09-21 历史暗谱，仅用于历史回放；真机采集前换成与你的测量设置匹配的暗谱。**
- `intersect.brokers[0].password`：替换 `CHANGE_ME`，与消息代理密码一致。
- 当前只用消息内的小型 JSON，未配置 MinIO；SDK 提示无法支持 MINIO 是预期信息。

服务启动时加载一次模型和暗谱。更换后重启服务。每个已绑定会话记录模型/暗谱 SHA256，不将新版本混入旧会话。
模型得分不是正确率；背景及低分返回 UNKNOWN，低质量/饱和谱返回 REJECTED。

先执行不连接设备的检查：

```powershell
.\.venv\Scripts\python.exe service.py --check
.\.venv\Scripts\python.exe replay.py '..\build\Desktop_Qt_6_6_3_MSVC2019_64bit-Debug\scan_sessions\1d_y_scan_20260921_152840_289'
```

### 2. 启动消息代理

这台机器检查时没有在 PATH 中发现 Docker。若已有可访问的 MQTT5 消息代理，可直接填写其地址；否则先准备 Docker Desktop，启动后运行：

```powershell
$env:RAMAN_BROKER_PASSWORD = '填写与config.json一致的密码'
docker compose up -d
```

Compose 默认仅向本机开放 1883。不要保留示例密码。

### 3. 打开桥接版 Qt

先关闭原 Qt 控制窗口，再运行：

```powershell
.\start_qt.cmd
```

此操作会像原 Qt 一样启动真实 Raman 后端；本次开发测试没有执行它。
在本地确认设备状态、完成激光开启和暗谱准备。需要远程扫描时勾选新增的 `Enable remote scans (existing travel paths)`。
此复选框默认关闭，重启 Qt 后需要重新启用。

### 4. 启动 Python 服务

另开 PowerShell，进入本目录：

```powershell
.\.venv\Scripts\python.exe service.py
```

服务保持运行。终止 Python 服务不会自动停止 Qt 的本地扫描，也不会自动关闭激光；这是第一版的既定行为。
网络中断时本地扫描可继续保存，恢复后查询状态和结果；不要无条件重新发送新的启动任务。

### 5. 用客户端操作

另开一个 PowerShell：

```powershell
.\.venv\Scripts\python.exe client.py status
.\.venv\Scripts\python.exe client.py start-scan --mode 1d-y --request-id experiment-001
```

启动成功返回 `session_id`。启动请求只等待 Qt 接受任务，不等待整次扫描完成。
下面用实际返回的会话编号替换 `SESSION_ID`：

```powershell
.\.venv\Scripts\python.exe client.py status
.\.venv\Scripts\python.exe client.py results SESSION_ID
.\.venv\Scripts\python.exe client.py cancel-scan SESSION_ID --request-id cancel-001
```

每页最多 50 点，下一页传上次返回的 `next_after_point`：

```powershell
.\.venv\Scripts\python.exe client.py results SESSION_ID --after-point 50
```

重复相同启动请求必须沿用同一个 `--request-id` 和参数。程序返回已有响应，不再次启动扫描。
如果请求超时，先查询状态，或者用原 request_id 重试。Qt 重启前遗留的未执行命令不会在新实例执行。
采集状态为 completed 时，SVM 可能仍在分析；查看 `analyzed_points`、`last_completed_point` 和 `analysis_error`。
第一版自动分析由 INTERSECT 成功提交并绑定的扫描。本地按钮独立启动的扫描可用 `replay.py` 分析。

## 另一台电脑

只需复制 `client.py`、`common.py` 和一份配置文件，并安装相同版本的 `intersect-sdk`；不需要复制 Qt、模型或厂商 DLL。
客户端实际只使用配置里的 `intersect` 部分，可以删除其他本机路径字段。
两台电脑的 hierarchy 必须一致，brokers 地址指向同一个消息代理。
如果使用这里的 Compose，将端口绑定由 `127.0.0.1` 改为设备电脑指定的局域网地址，并将防火墙范围限制到实验网络。
当前示例是实验局域网明文 MQTT；跨不可信网络应配置 TLS 或通过受控 VPN 连接。

## 重建与测试

```powershell
.\build_qt.cmd
.\build_qt.cmd --mock
.\.venv\Scripts\python.exe -m unittest test_bridge -v
.\.venv\Scripts\python.exe test_mock_integration.py
```

模拟测试只运行 `build_mock` 下无厂商 SDK 的测试程序。真实设备版不会被自动启动。
生成器只读原源码；若原源码结构变化导致锚点不唯一，会拒绝生成，需要检查后更新生成器。
新电脑部署可创建独立环境：`python -m venv .venv`，然后 `.venv\Scripts\python.exe -m pip install -r requirements.txt`。
`build_qt.cmd` 内的 Qt/Visual Studio 路径是针对这台电脑的，换电脑需调整。

## 已知限制

- 第一版接纯分类 SVM，不接 CNN/Transformer，也不做混合物识别或自适应选点。
- 保留现有扫描路径、采集超时和本地激光控制行为。
- Qt 文件邮箱仅是同机桥接，不应把其目录作为跨电脑共享命令通道；跨电脑使用 INTERSECT。
- 状态心跳超过 10 秒会报告离线/不确定；此状态不代表物理电机已经停止。
- INTERSECT 服务启动前必须配置可用模型和暗谱；模型不可用时服务拒绝启动。
- 设备版真实运行、消息代理容器和两台电脑通信尚需现场验证。

官方参考：
- https://github.com/ORNL/intersect-architecture
- https://github.com/INTERSECT-SDK/python-sdk
- https://intersect-python-sdk.readthedocs.io/en/latest/examples/hello-world.html
