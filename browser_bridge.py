#!/usr/bin/env python3
"""
Browser automation bridge using Playwright.
Playwright is optional - falls back to HTTP client if not available.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import Playwright, but make it optional
try:
    from playwright.sync_api import sync_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not installed. Browser automation disabled.")


class BrowserBridge:
    """
    Bridge for browser automation using Playwright (optional).
    Uses environment variables for configuration:
    - CHROME_EXECUTABLE: Path to Chrome/Chromium executable
    - CHROME_USER_DATA: Path to user data directory for persistent context
    """
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
        
        if not PLAYWRIGHT_AVAILABLE:
            logger.info("Playwright not available - browser bridge disabled")
            return
        
        # Get Chrome paths from environment variables
        self.chrome_executable = os.environ.get("CHROME_EXECUTABLE")
        self.chrome_user_data = os.environ.get("CHROME_USER_DATA")
        
        if not self.chrome_executable:
            logger.info("CHROME_EXECUTABLE not set - will use Playwright's default Chromium")
    
    def launch(self) -> bool:
        """Launch browser if Playwright is available."""
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Cannot launch browser: Playwright not installed")
            return False
        
        try:
            self.playwright = sync_playwright().start()
            launch_options = {"headless": self.headless}
            
            # Use custom Chrome executable if provided
            if self.chrome_executable and os.path.exists(self.chrome_executable):
                launch_options["executable_path"] = self.chrome_executable
                logger.info(f"Using Chrome executable: {self.chrome_executable}")
            
            self.browser = self.playwright.chromium.launch(**launch_options)
            
            # Use persistent context if user data dir provided
            if self.chrome_user_data:
                context = self.browser.new_context(user_data_dir=self.chrome_user_data)
                self.page = context.new_page()
                logger.info(f"Using user data dir: {self.chrome_user_data}")
            else:
                self.page = self.browser.new_page()
            
            logger.info("Browser launched successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            return False
    
    def close(self):
        """Close browser and cleanup."""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        logger.info("Browser closed")
    
    def navigate(self, url: str) -> bool:
        """Navigate to URL."""
        if not self.page:
            logger.error("Browser not launched")
            return False
        try:
            self.page.goto(url)
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False
    
    def get_content(self) -> Optional[str]:
        """Get page content."""
        if not self.page:
            return None
        try:
            return self.page.content()
        except Exception as e:
            logger.error(f"Failed to get content: {e}")
            return None


# For backward compatibility - HTTP endpoint already used via DeepSeekProxyClient
# This class now has no hardcoded paths and no auto-download logic.
# Playwright remains optional; if not installed, browser bridge is disabled.
