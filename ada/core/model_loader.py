import logging
from typing import Optional

try:
    from nexus_router import select_model
except ImportError:
    raise ImportError("nexus-router is required for dynamic model selection. Please install it.")

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles dynamic model selection via nexus-router."""

    @staticmethod
    def load_model(model_name: Optional[str] = None, **kwargs) -> str:
        """
        Load a model using nexus-router.select_model() for dynamic selection.

        Args:
            model_name: Optional model name hint. If None, nexus-router will decide.
            **kwargs: Additional parameters passed to select_model.

        Returns:
            The selected model identifier.
        """
        if model_name:
            logger.info(f"Using provided model hint: {model_name}")
        else:
            logger.info("No model name provided, delegating to nexus-router.select_model()")

        selected = select_model(model_name=model_name, **kwargs)
        logger.info(f"nexus-router selected model: {selected}")
        return selected

    @staticmethod
    def get_capabilities(model_id: Optional[str] = None) -> dict:
        """
        Get capabilities of the selected model.
        
        Args:
            model_id: Optional model ID. If None, select_model() is called first.

        Returns:
            Dictionary of model capabilities.
        """
        if model_id is None:
            model_id = ModelLoader.load_model()
        # Assumes nexus-router provides a get_capabilities function; adjust as needed.
        try:
            from nexus_router import get_model_capabilities
            return get_model_capabilities(model_id)
        except ImportError:
            logger.warning("get_model_capabilities not available in nexus-router")
            return {}
