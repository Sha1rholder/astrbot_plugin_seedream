"""Connect AstrBot lifecycle hooks to the image generation service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

from .src.generate import SeedreamAPI
from .src.service import ImageService

PLUGIN_NAME = "astrbot_plugin_seedream"


class SeedreamPlugin(Star):
    """Connect AstrBot lifecycle hooks to the configured image service."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        """Connect lifecycle hooks to an independent Seedream service.

        Args:
            context: AstrBot plugin context.
            config: Seedream plugin configuration.
        """
        super().__init__(context)
        plugin_dir = Path(__file__).resolve().parent
        self.service = ImageService(
            context,
            SeedreamAPI(config, plugin_dir),
            StarTools.get_data_dir(PLUGIN_NAME),
            self.logger,
        )

    @filter.command("seedream_help")
    async def seedream_help(
        self, event: AstrMessageEvent
    ) -> AsyncIterator[MessageEventResult]:
        """Show instructions after checking access and credentials.

        Args:
            event: Message requesting help.

        Yields:
            The guide or an actionable access or configuration error.
        """
        yield event.plain_result(self.service.instructions(event))

    @filter.on_llm_request()
    async def add_image_hint(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Append Seedream tool guidance to the current LLM request.

        Args:
            event: Original message event.
            req: Provider request being prepared.
        """
        self.service.add_image_hint(event, req)

    @filter.on_agent_done()
    async def mark_agent_finished(
        self,
        event: AstrMessageEvent,
        _context: ContextWrapper[AstrAgentContext],
        _response: LLMResponse,
    ) -> None:
        """Record agent completion without releasing unsent final text.

        Args:
            event: Event whose agent has finished.
            _context: Agent context supplied by AstrBot.
            _response: Final model response supplied by AstrBot.
        """
        await self.service.mark_agent_finished(event)

    @filter.after_message_sent()
    async def release_image_results(self, event: AstrMessageEvent) -> None:
        """Release images after the complete foreground reply.

        Args:
            event: Event whose message has been sent.
        """
        await self.service.release_image_results(event)

    async def terminate(self) -> None:
        """Cancel pending work and close the HTTP client."""
        await self.service.terminate()
