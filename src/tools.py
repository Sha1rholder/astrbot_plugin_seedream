"""Expose instruction and generation tools backed by the shared image service."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

from .generate import INPUT_SCHEMA, ImageGenerationError

if TYPE_CHECKING:
    from .service import ImageService

INSTRUCTIONS_TOOL_DESCRIPTION = "Check API key configuration, then read image generation instructions, local reference paths, HTTPS image URLs and available attachment indices."


class InstructionsTool(FunctionTool[AstrAgentContext]):
    """Provide Seedream instructions and available image references."""

    def __init__(self, service: ImageService) -> None:
        """Register the Seedream instructions tool.

        Args:
            service: Service providing access checks and reference discovery.
        """
        super().__init__(
            name="get_seedream_image_generate_instructions",
            description=INSTRUCTIONS_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
        self.service = service
        self.guide = (
            Path(__file__).with_name("instructions.md").read_text(encoding="utf-8")
        )

    def instructions(self, event: AstrMessageEvent) -> str:
        """Check credentials and append archive location and attachment indices.

        Args:
            event: Message supplying the available reference images.

        Returns:
            Instructions or a missing-credential explanation.
        """
        try:
            self.service.api.require_api_key()
        except ImageGenerationError as exc:
            return str(exc)
        images = self.service.reference_images(event)
        available = ", ".join(str(i) for i in range(len(images))) or "none"
        return (
            f"{self.guide}\n\nImage archive directory on the AstrBot host: {self.service.output_dir.resolve()}"
            f"\nAvailable reference image indices for this message: {available}"
        )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **_kwargs: Any
    ) -> str:
        """Return instructions after the service checks access.

        Args:
            context: Agent context containing the original message.
            **_kwargs: Unused arguments supplied by the tool executor.

        Returns:
            The guide or a configuration or access error.
        """
        return self.service.instructions(context.context.event)


class GenerateImageTool(FunctionTool[AstrAgentContext]):
    """Submit one Seedream image request through the plugin service."""

    def __init__(self, service: ImageService) -> None:
        """Register the generation tool.

        Args:
            service: Service handling admission, generation and delivery.
        """
        super().__init__(
            name="seedream_image_generate",
            description="Generate or edit one PNG image using Seedream 5.5. Read get_seedream_image_generate_instructions before use.",
            parameters=INPUT_SCHEMA,
        )
        self.service = service

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> str:
        """Forward the current event and original arguments to the service.

        Args:
            context: AstrBot agent context containing the original message.
            **kwargs: Generation arguments supplied by the agent.

        Returns:
            Transmission acknowledgement or a pre-submission error.
        """
        session = context.context.agent_session
        return await self.service.generate(
            context.context.event,
            kwargs,
            agent_session_id=session.session_id if session is not None else None,
        )
