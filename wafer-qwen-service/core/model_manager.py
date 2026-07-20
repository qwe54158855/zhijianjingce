import json
import logging
import os
import subprocess
import sys
import time
from typing import Optional

import httpx

from core.config import settings

logger = logging.getLogger(__name__)


class LlamaClient:
    """与 llama.cpp server 通信的 HTTP 客户端"""

    def __init__(self):
        self.base_url = settings.llama_server_url.rstrip("/")
        self.timeout = settings.llama_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.server_process = None
        self._ensure_server()

    def _ensure_server(self):
        """确保 llama-server 在运行（如果通过 URL 可访问则跳过）"""
        # 如果配置使用外部服务器，不自动启动
        if "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            return
        # 启动 WSL 中的 llama-server
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url.replace('/v1','')}/health")
            urllib.request.urlopen(req, timeout=2)
            logger.info("llama-server already running")
            return
        except Exception:
            logger.info("Starting llama-server via WSL...")

        model_path = settings.llama_model_path
        if not os.path.exists(model_path):
            # 尝试从 WSL 路径加载
            wsl_model = model_path.replace("D:", "/mnt/d").replace("\\", "/").replace("d:", "/mnt/d")
            model_path = wsl_model

        mmproj_path = settings.llama_mmproj_path
        if not os.path.exists(mmproj_path):
            mmproj_path = mmproj_path.replace("D:", "/mnt/d").replace("\\", "/").replace("d:", "/mnt/d")

        llama_bin = r"D:\cy\llama-b10068\llama-server.exe"
        if not os.path.exists(llama_bin):
            # Fallback: try WSL path
            cmd = [
                "wsl.exe", "bash", "-c",
                f"export LD_LIBRARY_PATH=/tmp/llama-bin/build/bin && exec /tmp/llama-bin/build/bin/llama-server "
                f"-m {model_path} "
                f"--mmproj {mmproj_path} "
                f"--port {self.base_url.split(':')[-1].split('/')[0] or '8002'} "
                f"--host 0.0.0.0 --ctx-size 4096 --batch-size 256 --n-gpu-layers 0"
            ]
        else:
            cmd = [
                llama_bin,
                "-m", model_path,
                "--mmproj", mmproj_path,
                "--port", self.base_url.split(':')[-1].split('/')[0] or '8002',
                "--host", "0.0.0.0",
                "--ctx-size", "4096",
                "--batch-size", "256",
                "--n-gpu-layers", "0",
            ]
        logger.info(f"Launching: {' '.join(cmd)}")
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            logger.info(f"llama-server started, PID: {self.server_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")

    async def check_health(self) -> bool:
        """检查 llama-server 是否可用"""
        try:
            r = await self.client.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception as e:
            logger.warning(f"llama-server health check failed: {e}")
            return False

    async def analyze_image(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Optional[dict]:
        """
        调用 Qwen-VL 分析图像并返回结构化 JSON。
        """
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            r = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]

            # 从回复中提取 JSON
            content = content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Qwen response as JSON: {e}\nResponse: {content}")
            return None
        except Exception as e:
            logger.error(f"Qwen inference failed: {e}")
            return None

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """纯文本聊天（不带图片）"""
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            r = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return None

    async def generate_report(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """调用 Qwen-VL 生成文本报告"""
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            r = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None

    async def close(self):
        await self.client.aclose()
