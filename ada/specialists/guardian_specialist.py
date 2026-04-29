"""Guardian Specialist for Ada.

This specialist communicates with the ada-guardian API to check system resources
before invoking the primary Ollama model. It ensures that resource constraints
(e.g., memory, CPU, GPU availability) are met before proceeding with model calls.
"""

import logging
import requests
from typing import Dict, Any, Optional

from ada.specialists.base_specialist import BaseSpecialist

logger = logging.getLogger(__name__)


class GuardianSpecialist(BaseSpecialist):
    """Specialist that checks system resources via ada-guardian API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.guardian_url = self.config.get("guardian_url", "http://localhost:8080")
        self.timeout = self.config.get("timeout", 5)
        self.required_resources = self.config.get("required_resources", {})

    def check_resources(self) -> Dict[str, Any]:
        """Query ada-guardian for current system resource status.

        Returns:
            Dictionary containing resource metrics and availability status.
        """
        try:
            response = requests.get(
                f"{self.guardian_url}/api/v1/resources",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to communicate with ada-guardian: {e}")
            return {"error": str(e), "available": False}

    def is_resources_sufficient(self, resources: Dict[str, Any]) -> bool:
        """Check if available resources meet required thresholds.

        Args:
            resources: Resource data from ada-guardian.

        Returns:
            True if sufficient, False otherwise.
        """
        if "error" in resources:
            return False

        for resource, required in self.required_resources.items():
            available = resources.get(resource)
            if available is None:
                logger.warning(f"Resource {resource} not reported by guardian")
                return False
            if available < required:
                logger.warning(
                    f"Insufficient {resource}: available={available}, required={required}"
                )
                return False
        return True

    def before_model_invoke(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called before invoking the primary Ollama model.

        Args:
            context: Current execution context.

        Returns:
            Updated context with resource check results.
        """
        logger.info("Checking system resources via ada-guardian...")
        resources = self.check_resources()
        context["guardian_resources"] = resources

        if not self.is_resources_sufficient(resources):
            logger.error("Insufficient resources for model invocation")
            context["guardian_blocked"] = True
            context["guardian_message"] = "Resource constraints prevent model invocation"
        else:
            logger.info("Resources sufficient, proceeding with model invocation")
            context["guardian_blocked"] = False

        return context

    def can_proceed(self, context: Dict[str, Any]) -> bool:
        """Determine if primary model can be invoked.

        Args:
            context: Context containing guardian check results.

        Returns:
            True if resources are sufficient, False otherwise.
        """
        return not context.get("guardian_blocked", False)

    def get_specialty_name(self) -> str:
        """Return the name of this specialist."""
        return "guardian"
