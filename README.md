# MoxhiMT Runner & API

Project Python chạy cục bộ hai mô hình [DanVP/MoxhiMT-30](https://huggingface.co/DanVP/MoxhiMT-30) và [DanVP/MoxhiMT-60](https://huggingface.co/DanVP/MoxhiMT-60) để dịch văn bản **tiếng Trung sang tiếng Việt**. Có CLI và API server nhận/trả raw text.

> MoxhiMT-30 dùng giấy phép **CC-BY-NC-4.0** (phi thương mại); MoxhiMT-60 dùng **Apache-2.0**. Hãy kiểm tra và biên tập lại kết quả trước khi xuất bản.

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[pytorch]'
```

Lần chạy đầu tiên sẽ tải model từ Hugging Face và lưu trong cache cục bộ. Các lần sau có thể chạy offline nếu cache vẫn còn. Extra `pytorch` phục vụ CLI Transformers; server CPU production dùng extra `cpu` nhẹ hơn.

## API server

Ba entrypoint được tách riêng theo runtime:

| File | Runtime | Backend |
|---|---|---|
| `server_cpu.py` | CPU | CTranslate2 INT8 |
| `server_cuda.py` | NVIDIA CUDA | PyTorch Transformers |
| `server_rocm.py` | AMD ROCm/HIP | PyTorch Transformers |

CUDA và ROCm dùng chung toàn bộ implementation GPU. PyTorch ROCm cũng gọi thiết bị GPU là `cuda`; entrypoint riêng có nhiệm vụ kiểm tra đúng loại wheel/runtime lúc khởi động.

### Cài đặt và chạy bằng script

CPU:

```bash
./scripts/install.sh cpu
./scripts/run-server.sh cpu --host 0.0.0.0 --port 8000
```

NVIDIA CUDA:

```bash
./scripts/install.sh cuda
./scripts/run-server.sh cuda --host 0.0.0.0 --port 8000
```

AMD ROCm:

```bash
./scripts/install.sh rocm
./scripts/run-server.sh rocm --host 0.0.0.0 --port 8000
```

Mặc định script tạo `.venv` trong project. Có thể đổi đường dẫn hoặc Python:

```bash
VENV_DIR=/opt/moxhimt/venv PYTHON_BIN=python3.12 ./scripts/install.sh cpu
VENV_DIR=/opt/moxhimt/venv ./scripts/run-server.sh cpu
```

Trên Debian/Ubuntu, nếu Python chưa có module tạo virtualenv, cài trước `python3-venv`.

Script CUDA mặc định dùng wheel index `cu128`; ROCm mặc định dùng `rocm7.1`. Nếu driver/runtime production yêu cầu bản khác, truyền index chính thức phù hợp:

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 ./scripts/install.sh cuda
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.0 ./scripts/install.sh rocm
```

Installer kiểm tra GPU sau khi cài. Khi chỉ chuẩn bị environment trên build machine không có GPU, có thể bỏ qua riêng bước kiểm tra thiết bị:

```bash
SKIP_DEVICE_CHECK=1 ./scripts/install.sh rocm
```

Server preload cả hai model trước khi readiness trả về thành công. Hai endpoint nhận **toàn bộ request body UTF-8** và trả về `text/plain; charset=utf-8`:

```bash
curl --fail-with-body \
  -H 'Content-Type: text/plain; charset=utf-8' \
  --data-binary '他抬头望向远处的山门。' \
  http://127.0.0.1:8000/moxhi-30

curl --fail-with-body \
  -H 'Content-Type: text/plain; charset=utf-8' \
  --data-binary @truyen-zh.txt \
  http://127.0.0.1:8000/moxhi-60 > truyen-vi.txt
```

Với CUDA/ROCm, mỗi request được dịch bằng một lần gọi model trên toàn bộ các dòng không rỗng. Các request ngắn cùng model được gom vào một lần gọi GPU khi đạt 100 dòng, hoặc khi hết cửa sổ gom 300 ms tính từ request đầu tiên của nhóm. Nếu GPU đang bận, request vẫn chờ đến lượt theo `MOXHI_QUEUE_TIMEOUT_SECONDS`. Request dài hơn 100 dòng được chạy riêng ngay khi tới lượt, không bị chia nhỏ. CPU vẫn chia theo `MOXHI_BATCH_SIZE`.

Health checks:

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

### Benchmark generated tokens/giây

Sửa nội dung cần đo trong `sample-input.txt`, khởi động server rồi chạy:

```bash
python benchmark.py --model 30
python benchmark.py --model 60
```

Mặc định script warm-up một lượt, đo năm lượt và gọi server tại `http://127.0.0.1:8000`. Có thể thay đổi:

```bash
python benchmark.py \
  --model 60 \
  --url http://127.0.0.1:8000 \
  --input sample-input.txt \
  --warmup 2 \
  --runs 10
```

Kết quả gồm generated tokens/giây từng lượt, throughput tổng, min/max và latency trung bình, p50, p95. Token đầu ra được đếm bằng đúng tokenizer của model; mỗi dòng dịch không rỗng cộng thêm một token EOS thực sự được generate. Thời gian model load và warm-up không nằm trong kết quả đo, còn HTTP overhead được tính vì đây là benchmark end-to-end của API.

### So sánh performance giữa các thiết bị

`benchmark-devices.py` load model và gọi trực tiếp CTranslate2/PyTorch, hoàn toàn không khởi động API server. Input được tokenize trước khi đo; thời gian tải model, tokenization, HTTP và decode text không nằm trong throughput. Với GPU, script synchronize thiết bị trước khi kết thúc mỗi lượt đo.

```bash
# CPU
.venv/bin/python benchmark-devices.py cpu --model 60

# Từng GPU NVIDIA, chạy tuần tự card 0, 1 và 2
.venv/bin/python benchmark-devices.py cuda --devices 0,1,2 --model 60

# Từng GPU AMD ROCm
.venv/bin/python benchmark-devices.py rocm --devices 0,1 --model 60
```

Mặc định mỗi device warm-up 2 lượt và đo 10 lượt. Để chạy lâu hơn nhằm tìm throughput cao nhất:

```bash
.venv/bin/python benchmark-devices.py cuda \
  --devices 0,1 \
  --model 30 \
  --warmup 5 \
  --runs 50
```

Mặc định nội dung `sample-input.txt` được lặp 16 lần để workload đủ lớn, tránh kết quả bị chi phối bởi overhead của một batch quá nhỏ. Điều chỉnh bằng `--repeat` và `--batch-size` nếu cần. Mặc định dịch greedy (`--num-beams 1`); truyền `--num-beams 4` nếu muốn chạy beam search. Các GPU chạy tuần tự theo index nên không tranh tài nguyên với nhau. Script báo `MAX generated tokens/s`, độ trễ và throughput trung vị của từng device; dòng cuối là device nhanh nhất. Hãy chạy script bằng virtualenv đã được `install.sh` cài đúng runtime của máy.

Để so sánh F32 và F16 trên ROCm với toàn bộ các dòng không rỗng trong một batch, đặt `--repeat 1` và `--batch-size` bằng số dòng không rỗng. Ví dụ với `sample-input.txt` hiện có 344 dòng:

```bash
.venv/bin/python benchmark-devices.py rocm --model 30 --dtype f32 --repeat 1 --batch-size 344
.venv/bin/python benchmark-devices.py rocm --model 30 --dtype f16 --repeat 1 --batch-size 344
```

CPU dùng bản CTranslate2 INT8 của model. Backend này không chạy trên ROCm; kết quả CPU INT8 được ghi riêng theo thiết bị. Script cũng in độ trễ và throughput trung vị của các lượt đo sau warm-up.

### Thử nghiệm Rust/ONNX Runtime

Cargo project độc lập tại [`rust-onnx/`](rust-onnx/README.md) chạy các file ONNX fp16 có sẵn của `DanVP/MoxhiMT-30-web` trên CPU hoặc CUDA. Nó dịch greedy theo batch và đo generated tokens/s, dùng cùng `sample-input.txt` nếu không truyền `--text` hoặc `--input`. Xem hướng dẫn tải model, cấu hình thư viện ONNX Runtime và lệnh benchmark trong README riêng. Đây là đường chạy ONNX Runtime; trang `moxhi.pages.dev` sử dụng bộ giải mã WebGPU riêng.

### Cấu hình production

Server đọc các biến môi trường sau:

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `MOXHI_30_MODEL` | `DanVP/MoxhiMT-30` | ID/đường dẫn model endpoint 30 |
| `MOXHI_60_MODEL` | `DanVP/MoxhiMT-60` | ID/đường dẫn model endpoint 60 |
| `MOXHI_30_REVISION` | trống | Revision cố định của model 30 |
| `MOXHI_60_REVISION` | trống | Revision cố định của model 60 |
| `MOXHI_MAX_BODY_BYTES` | `2097152` | Kích thước request tối đa |
| `MOXHI_CONCURRENCY_PER_MODEL` | `1` | Số lượt suy luận đồng thời cho mỗi model CPU; GPU dùng một queue theo model |
| `MOXHI_QUEUE_TIMEOUT_SECONDS` | `30` | Thời gian chờ slot trước khi trả 503 |
| `MOXHI_BATCH_SIZE` | `8` | Số dòng trong một batch CPU; GPU xử lý cả request cùng lúc |
| `MOXHI_GPU_BATCH_MAX_LINES` | `100` | Mục tiêu số dòng tối đa khi gom các request GPU ngắn |
| `MOXHI_GPU_BATCH_WAIT_MS` | `300` | Thời gian chờ gom tối đa, từ 0 đến 300 ms |
| `MOXHI_NUM_BEAMS` | `1` | Số beam khi sinh bản dịch (`4` nếu muốn beam search) |

Có thể chạy trực tiếp bằng console entrypoint thay cho bash script:

```bash
moxhimt-server-cpu --host 0.0.0.0 --port 8000
moxhimt-server-cuda --host 0.0.0.0 --port 8000
moxhimt-server-rocm --host 0.0.0.0 --port 8000
```

Các entrypoint cố định backend/device nên `MOXHI_BACKEND` và `MOXHI_DEVICE` không được dùng với ba server chuyên biệt. Chỉ chạy **một worker trên mỗi GPU/process** vì mỗi worker tải riêng cả hai model. Scale bằng nhiều container/replica phía sau reverse proxy thay vì tăng `--workers` trong cùng container.

### Docker CPU

```bash
docker build -t moxhimt-api .
docker run --rm -p 8000:8000 \
  -v moxhimt-cache:/models/huggingface \
  moxhimt-api
```

Volume cache tránh tải lại model sau mỗi lần tạo container.

## Sử dụng

Dịch một câu:

```bash
moxhimt --text "他抬头望向远处的山门。"
```

Dịch từ stdin:

```bash
echo "今天天气很好。" | moxhimt
```

Dịch tệp UTF-8 và giữ nguyên các dòng trống:

```bash
moxhimt --input truyen-zh.txt --output truyen-vi.txt
```

CLI tự chọn CUDA, Apple MPS hoặc CPU. Có thể chỉ định thiết bị:

```bash
moxhimt --device cuda --text "你好，世界！"
moxhimt --device cpu --text "你好，世界！"
```

### Chạy CPU nhanh bằng CTranslate2 INT8

Model cung cấp sẵn bản INT8, phù hợp với máy không có GPU:

```bash
python -m pip install -e '.[cpu]'
moxhimt --backend ct2-int8 --input truyen-zh.txt --output truyen-vi.txt
```

### Cố định phiên bản model

Mặc định dùng nhánh `main` mới nhất. Để kết quả có thể tái lập, đặt revision cụ thể:

```bash
moxhimt --revision v4.0.3 --text "他抬头望向远处的山门。"
```

Các tùy chọn sinh mặc định: `--num-beams 1`, `--max-source-length 512`, `--max-length 512`. Có thể chọn `--num-beams 4` khi muốn beam search theo khuyến nghị trên model card. Mỗi dòng không trống là một đơn vị dịch; nếu một dòng vượt giới hạn token, CLI báo lỗi để tránh âm thầm cắt mất văn bản. Hãy tách đoạn quá dài thành nhiều câu/dòng. Xem toàn bộ tùy chọn bằng `moxhimt --help`.

## Dùng trong Python

```python
from moxhimt_runner import Translator

translator = Translator(device="auto")
print(translator.translate_document("他抬头望向远处的山门。"))
```

## Phát triển và kiểm thử

```bash
python -m pip install -e '.[pytorch,cpu,server,dev]'
ruff check .
pytest
```

Các test mặc định dùng backend giả, vì vậy không tải model và không cần mạng/GPU.
