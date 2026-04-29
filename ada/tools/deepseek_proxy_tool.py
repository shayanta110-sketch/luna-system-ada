#!/usr/bin/env python3
"""
DeepSeek Proxy Tool
Provides client and utilities to interact with local DeepSeek proxy server for cloud-based reasoning.
"""

import json
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek proxy connection."""
    base_url: str = "http://localhost:8000"
    timeout: int = 60
    max_retries: int = 3


class DeepSeekProxyClient:
    """Client for interacting with local DeepSeek proxy server."""

    def __init__(self, config: Optional[DeepSeekConfig] = None):
        self.config = config or DeepSeekConfig()
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(self, endpoint: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send request to proxy server."""
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(url, json=payload, timeout=self.config.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    print(f"DeepSeek proxy error: {e}")
                    return None
        return None

    def ask_deepseek(self, prompt: str, system_prompt: Optional[str] = None,
                     temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """
        Send a prompt to DeepSeek and get the response.

        Args:
            prompt: User's question or instruction
            system_prompt: Optional system message to set context
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum response length

        Returns:
            Response text or None if failed
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        result = self._request("v1/chat/completions", payload)
        if result and "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("message", {}).get("content")
        return None

    def stream_ask(self, prompt: str, system_prompt: Optional[str] = None,
                   temperature: float = 0.7, max_tokens: int = 2000):
        """
        Stream response from DeepSeek token by token.

        Yields:
            Chunks of response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        url = f"{self.config.base_url}/v1/chat/completions"
        try:
            with self.session.post(url, json=payload, stream=True, timeout=self.config.timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
        except requests.exceptions.RequestException as e:
            print(f"Streaming error: {e}")

    def is_available(self) -> bool:
        """Check if DeepSeek proxy server is reachable."""
        try:
            response = self.session.get(f"{self.config.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


# Convenience functions
default_client = DeepSeekProxyClient()


def ask_deepseek(prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
    """
    Convenience function to query DeepSeek proxy.

    Args:
        prompt: User's question or instruction
        system_prompt: Optional system message to set context
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum response length

    Returns:
        Response text or None if failed
    """
    return default_client.ask_deepseek(prompt, system_prompt, temperature, max_tokens)


def is_deepseek_available() -> bool:
    """
    Check if DeepSeek proxy server is available.

    Returns:
        True if proxy is reachable and responding, False otherwise
    """
    return default_client.is_available()


# Example usage
if __name__ == "__main__":
    if is_deepseek_available():
        response = ask_deepseek("Explain quantum computing in one sentence.")
        if response:
            print("DeepSeek response:", response)
        else:
            print("Failed to get response")
    else:
        print("DeepSeek proxy not available. Start it first.")
