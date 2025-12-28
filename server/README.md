# HTTP 接口

## 依赖安装

```bash
cd server
pip install -r requirements.txt
```

## 运行

```bash
# default run dev
make

# run dev
make dev

# run prod
make run
```

## 使用

```bash
curl -X POST "http://127.0.0.1:8000/uploadimage/" -H "accept: image/jpeg" -H "Content-Type: multipart/form-data" -F "file=@src-001.jpg" -F "scale=2" > dst-001.jpg
```
