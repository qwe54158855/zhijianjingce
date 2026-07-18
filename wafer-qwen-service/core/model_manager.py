import json
import logging
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
