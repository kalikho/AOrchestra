from __future__ import annotations

from typing import Any, Dict, Optional
import base64
import os
from pathlib import Path
from openai import AsyncOpenAI 
from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig


## Image Tools
class ImageAnalysisAction(BaseAction):
    name: str = "ImageAnalysisAction"
    description: str = (
        "Analyze a local image file using a multimodal model. "
        "Only supports local file paths (not URLs). "
        "If you need to analyze an online image, download it first with ExecuteCodeAction, then pass the local path."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Instruction or question describing what to analyze in the image",
            },
            "image_path": {
                "type": "string",
                "description": "Local file path to the image (e.g. '/path/to/image.png'). URLs are NOT supported.",
            }
        },
        "required": ["query", "image_path"],
        "additionalProperties": False,
    }
    
    DEFAULT_MODEL: str = "gemini-3-flash-preview"

    MAX_IMAGE_PIXELS: int = 1568 * 1568
    MAX_IMAGE_BYTES: int = 1 * 1024 * 1024  # 1MB

    def encode_image(self, image_path: str) -> str:
        path = Path(image_path).expanduser()
        data = path.read_bytes()

        if len(data) <= self.MAX_IMAGE_BYTES:
            return base64.b64encode(data).decode("utf-8")

        from PIL import Image
        import io
        img = Image.open(path)
        w, h = img.size
        if w * h > self.MAX_IMAGE_PIXELS:
            ratio = (self.MAX_IMAGE_PIXELS / (w * h)) ** 0.5
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        fmt = "JPEG" if path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
        img.save(buf, format=fmt, quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    
    def _get_llm_config(self, model_name: Optional[str] = None) -> tuple:
        """Get LLM configuration from model_config.yaml or environment variables."""
        model_name = model_name or self.DEFAULT_MODEL
        
        # Try to get config from LLMsConfig (model_config.yaml)
        try:
            llms_config = LLMsConfig.default()
            model_config = llms_config.get(model_name)
            if model_config:
                return (
                    model_config.key,
                    model_config.base_url,
                    model_config.model,
                )
        except Exception:
            pass
        
        # Fallback to environment variables
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://newapi.deepwisdom.ai/v1"
        
        return (api_key, base_url, model_name)

    async def __call__(self, **kwargs) -> Any:
        query = kwargs.get("query")
        image_path = kwargs.get("image_path")

        if not query or not image_path:
            return {"success": False, "output": None, "error": "Both query and image_path are required.", "metrics": {}}

        api_key, base_url, model = self._get_llm_config()
        if not api_key:
            return {"success": False, "output": None, "error": "No API key found. Configure in model_config.yaml or set OPENAI_API_KEY", "metrics": {}}

        try:
            from openai import AsyncOpenAI  # type: ignore
        except Exception:
            return {"success": False, "output": None, "error": "openai package not available", "metrics": {}}

        try:
            if image_path.startswith(("http://", "https://")):
                image_url = image_path
            else:
                encoded = self.encode_image(image_path)
                image_url = f"data:image/png;base64,{encoded}"
        except Exception as exc:
            return {"success": False, "output": None, "error": f"Failed to prepare image: {exc}", "metrics": {}}

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": str(query)},
                ],
            }
        ]

        try:
            completion = await client.chat.completions.create(model=model, messages=messages)
        except Exception as exc:
            return {"success": False, "output": None, "error": f"Image analysis failed: {exc}", "metrics": {}}

        content = None
        try:
            if completion and completion.choices:
                content = completion.choices[0].message.content
        except Exception:
            content = None

        if not content:
            return {"success": False, "output": None, "error": "Model returned empty response", "metrics": {}}

        return {"success": True, "output": content.strip(), "error": None, "metrics": {}}

## Audio Tools
class ParseAudioAction(BaseAction):
    name: str = "ParseAudioAction"
    description: str = (
        "Transcribe and analyze a local audio file. "
        "Only supports local file paths (not URLs). "
        "If you need to analyze an online audio file, download it first with ExecuteCodeAction, then pass the local path. "
        "Step 1: Transcribes audio to text using Whisper. "
        "Step 2: Analyzes the transcript based on your query using an LLM."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Instruction that describes how to handle or summarize the audio content",
            },
            "audio_path": {
                "type": "string",
                "description": "Local file path that points to the audio resource (e.g. mp3, wav)",
            },
        },
        "required": ["query", "audio_path"],
        "additionalProperties": False,
    }
    
    DEFAULT_MODEL: str = "whisper"
    
    def _get_api_credentials(self) -> tuple:
        """Get API key and base_url from model_config.yaml (borrow from any configured model)."""
        try:
            llms_config = LLMsConfig.default()
            for name in ("gemini-3-flash-preview", "gpt-4o", "claude-4-sonnet"):
                try:
                    cfg = llms_config.get(name)
                    return cfg.key, cfg.base_url
                except (ValueError, KeyError):
                    continue
        except Exception:
            pass
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or "https://newapi.deepwisdom.ai/v1"
        return api_key, base_url

    async def __call__(self, **kwargs) -> Any:
        prompt = kwargs.get("query")
        audio_path = kwargs.get("audio_path")

        api_key, base_url = self._get_api_credentials()
        if not api_key:
            return {
                "success": False,
                "output": None,
                "error": "No API key found. Configure in model_config.yaml or set OPENAI_API_KEY",
                "metrics": {},
            }

        from openai import OpenAI as SyncOpenAI

        try:
            whisper_client = SyncOpenAI(api_key=api_key, base_url=base_url)
            with open(audio_path, "rb") as f:
                transcript_resp = whisper_client.audio.transcriptions.create(
                    model=self.DEFAULT_MODEL,
                    file=f,
                )
            transcript = transcript_resp.text.strip()

            analysis_model = "gemini-3-flash-preview"
            llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            try:
                cfg = LLMsConfig.default().get(analysis_model)
                if cfg:
                    llm_client = AsyncOpenAI(api_key=cfg.key, base_url=cfg.base_url)
                    analysis_model = cfg.model
            except Exception:
                pass

            completion = await llm_client.chat.completions.create(
                model=analysis_model,
                messages=[
                    {"role": "user", "content": f"Below is a transcript of an audio recording.\n\nTranscript:\n{transcript}\n\nInstruction: {prompt}"}
                ],
            )
            answer = completion.choices[0].message.content.strip()
            return {"success": True, "output": answer, "error": None, "metrics": {"transcript": transcript}}
        except Exception as exc:
            return {"success": False, "output": None, "error": f"Audio processing failed: {exc}", "metrics": {}}