import json
from typing import List, Dict, Any
from langchain.tools import BaseTool

from ada.tools.steno_compressor import StenoCompressor


class CompressConversationTool(BaseTool):
    """Tool to compress conversation exchanges using StenoCompressor."""

    name: str = "compress_conversation"
    description: str = (
        "Compresses a list of conversation exchanges (each exchange should be a dict "
        "with 'role' and 'content') and returns the compressed string along with the "
        "compression ratio."
    )

    def _run(self, exchanges: List[Dict[str, str]]) -> str:
        """Execute the compression.

        Args:
            exchanges: List of conversation exchanges, e.g.
                [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]

        Returns:
            A JSON string containing compressed text and compression ratio.
        """
        compressor = StenoCompressor()
        compressed_str = compressor.compress(exchanges)
        original_str = json.dumps(exchanges)
        ratio = len(compressed_str) / len(original_str) if original_str else 1.0

        result = {
            "compressed_text": compressed_str,
            "compression_ratio": round(ratio, 4),
            "original_length": len(original_str),
            "compressed_length": len(compressed_str),
        }
        return json.dumps(result, indent=2)

    async def _arun(self, exchanges: List[Dict[str, str]]) -> str:
        """Async version of _run."""
        return self._run(exchanges)
