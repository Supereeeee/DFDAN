import os
import tempfile
import logging
from uuid import uuid4
from mimetypes import guess_extension
from typing import Union

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse

from server.scale import sacle

app = FastAPI()

logging.basicConfig(level=logging.INFO)


@app.get("/")
def read_root():
    return {"Hello": "DFDAN"}


# 文件上传的路由
# 接收上传的图片，然后调用 inference/inference_DFDAN.py 中的 main 函数进行推理,返回图片，二进制流
@app.post("/uploadimage/")
async def upload_image(
    scale: Union[int, None] = Form(default=2), file: UploadFile = File(...)
):
    # 参数注释
    """
    Upload an image to be scaled by DFDAN.

    Parameters:
    - scale: int, optional, default 2, scale factor for the super-resolution model, must be 2, 3, or 4.
    - file: UploadFile, required, the image file to be scaled.

    Returns:
    - StreamingResponse, the scaled

    Raises:
    - HTTPException: if the scale factor is not 2, 3, or 4.
    - HTTPException: if the uploaded file is not an image.

    Examples:
    - Upload an image and scale it by 2x:
    ```
    curl -X POST "http://127.0.0.1:8000/uploadimage/" -H "accept: image/jpeg" -H "Content-Type: multipart/form-data" -F "file=@src-001.jpg" > dst-001.jpg
    ```
    - Upload an image and scale it by 4x:
    ```
    curl -X POST "http://127.0.0.1:8000/uploadimage/" -H "accept: image/jpeg" -H "Content-Type: multipart/form-data" -F "file=@src-001.jpg" -F "scale=4" > dst-002.jpg
    ```

    """
    # 检查 scale 是否为 2, 3, 4，若不是则返回错误提示
    if scale not in [2, 3, 4]:
        raise HTTPException(status_code=400, detail="Scale must be 2, 3, 4.")

    # async def upload_image(file: bytes = File(...)):
    # 检查文件类型是否为图片
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # 获取文件的内容
    file_contents = await file.read()

    # 获取文件的MIME类型
    mime_type = file.content_type

    # 根据MIME类型获取文件的扩展名
    guessed_extension = guess_extension(mime_type)

    # 如果无法从MIME类型获取扩展名，则默认使用'.bin'
    if not guessed_extension:
        guessed_extension = ".bin"

    # 使用 uuid4 生成唯一的文件名
    filename = f"DFDAN_upload_{uuid4().hex}{guessed_extension}"

    # 指定临时目录
    # 保存上传的图片到临时目录，以留做备份
    # 使用 tempfile.gettempdir() 获取系统临时目录
    temp_file_path = os.path.join(tempfile.gettempdir(), filename)
    logging.info(f"Saving uploaded file to {temp_file_path}")
    # 将文件写入临时目录中
    with open(temp_file_path, "wb") as f:
        f.write(file_contents)

    src_path = temp_file_path
    logging.info(f"Uploaded file name: {src_path}")

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".png", prefix="DFDAN_scale_"
    ) as fp:
        dst_path = fp.name
        logging.info(f"Saving scaled file to {dst_path}")
        # 调用 scale.py 中的 sacle 函数进行推理
        sacle(src_path, dst_path, scale=2)

        def iterfile():
            with open(dst_path, mode="rb") as file_like:
                yield from file_like

        return StreamingResponse(iterfile(), media_type="image/png")
