"""
browser package

Exposes BrowserService as the single entry point for all browser automation.

    from browser import BrowserService

    service = BrowserService()
    service.start()
    service.open("https://example.com")
    service.screenshot("storage/screenshots/example.png")
    service.close()
"""

from browser.browser_service import BrowserService

__all__ = ["BrowserService"]
