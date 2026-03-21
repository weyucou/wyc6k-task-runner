"""DRF views for the agents app."""

import asyncio
import logging
from typing import Any

from django.db import transaction
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response

from agents.models import Agent, AgentCredential, AgentTool, Tool
from agents.runner import AgentRunner
from agents.serializers import (
    AgentCreateSerializer,
    AgentDetailSerializer,
    AgentListSerializer,
    AgentToolSerializer,
    ChatRequestSerializer,
    ChatResponseSerializer,
    MemorySearchRequestSerializer,
    MemorySearchResponseSerializer,
    ToolSerializer,
)

logger = logging.getLogger(__name__)


class AgentViewSet(viewsets.ModelViewSet):
    """ViewSet for Agent CRUD operations and chat."""

    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Return all agents."""
        return Agent.objects.prefetch_related("agent_tools__tool")

    def get_serializer_class(self):
        """Return appropriate serializer based on action."""
        if self.action == "list":
            return AgentListSerializer
        if self.action == "create":
            return AgentCreateSerializer
        return AgentDetailSerializer

    def perform_create(self, serializer: AgentCreateSerializer) -> None:
        """Create agent and store encrypted API key."""
        api_key = serializer.validated_data.pop("api_key", None)

        with transaction.atomic():
            agent = serializer.save()

            if api_key:
                # TODO: Implement proper encryption
                AgentCredential.objects.create(
                    agent=agent,
                    encrypted_api_key=api_key,
                )

    def perform_update(self, serializer: AgentDetailSerializer) -> None:
        """Update agent configuration."""
        serializer.save()

    @action(detail=True, methods=["post"])
    def chat(self, request: Request, pk: int | None = None) -> Response:
        """Send a message to the agent and get a response.

        Request body:
            message: str - The user message
            conversation_history: list[dict] - Optional prior messages
            enable_tools: bool - Whether to enable tool calling (default: True)
            tool_names: list[str] - Optional specific tools to enable
            system_prompt: str - Optional system prompt override

        Returns:
            Response with agent reply and metadata
        """
        agent = self.get_object()

        if not agent.is_active:
            return Response(
                {"error": "Agent is not active"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data

        try:
            # Run the async chat in a sync context
            result = asyncio.run(
                self._run_chat(
                    agent=agent,
                    message=data["message"],
                    conversation_history=data.get("conversation_history", []),
                    enable_tools=data.get("enable_tools", True),
                    tool_names=data.get("tool_names"),
                    system_prompt=data.get("system_prompt"),
                )
            )

            response_serializer = ChatResponseSerializer(result)
            return Response(response_serializer.data)

        except Exception:
            logger.exception("Chat error for agent %s", agent.name)
            return Response(
                {"error": "Failed to process chat request"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    async def _run_chat(
        self,
        agent: Agent,
        message: str,
        conversation_history: list[dict[str, str]],
        enable_tools: bool,
        tool_names: list[str] | None,
        system_prompt: str | None,
    ) -> dict[str, Any]:
        """Execute chat with the agent.

        Args:
            agent: The agent to chat with.
            message: User message.
            conversation_history: Prior conversation messages.
            enable_tools: Whether to enable tools.
            tool_names: Specific tools to enable.
            system_prompt: System prompt override.

        Returns:
            Chat response dictionary.
        """
        from agents.llm import LLMMessage  # noqa: PLC0415

        runner = AgentRunner(agent)

        # Build message history
        messages: list[LLMMessage] = []
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                messages.append(LLMMessage.system(content))
            elif role == "assistant":
                messages.append(LLMMessage.assistant(content))
            else:
                messages.append(LLMMessage.user(content))

        # Add the new user message
        messages.append(LLMMessage.user(message))

        # Apply tool profile filtering
        filtered_tools = tool_names
        if enable_tools and not tool_names:
            # Get available tools and filter by profile
            available_tools = runner.registry.list_tools()
            filtered_tools = agent.get_allowed_tools(available_tools)

        # Run the agent
        response, history = await runner.run(
            messages,
            system_prompt=system_prompt,
            enable_tools=enable_tools and bool(filtered_tools),
            tool_names=filtered_tools,
        )

        return {
            "content": response.content,
            "agent": agent.name,
            "model": response.model or agent.model_name,
            "stop_reason": response.stop_reason.value if response.stop_reason else "unknown",
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "tool_calls": [tc.to_dict() for tc in response.tool_calls] if response.tool_calls else [],
            "history": [msg.to_dict() for msg in history],
        }

    @action(detail=True, methods=["get"])
    def tools(self, request: Request, pk: int | None = None) -> Response:
        """Get the tools assigned to this agent."""
        agent = self.get_object()
        agent_tools = agent.agent_tools.select_related("tool").filter(is_enabled=True)
        serializer = AgentToolSerializer(agent_tools, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def add_tool(self, request: Request, pk: int | None = None) -> Response:
        """Add a tool to this agent.

        Request body:
            tool_id: int - ID of the tool to add
            config: dict - Optional tool configuration
        """
        agent = self.get_object()
        tool_id = request.data.get("tool_id")

        if not tool_id:
            return Response(
                {"error": "tool_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            tool = Tool.objects.get(id=tool_id, is_active=True)
        except Tool.DoesNotExist:
            return Response(
                {"error": "Tool not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        agent_tool, created = AgentTool.objects.get_or_create(
            agent=agent,
            tool=tool,
            defaults={
                "is_enabled": True,
                "config": request.data.get("config", {}),
            },
        )

        if not created:
            agent_tool.is_enabled = True
            agent_tool.config = request.data.get("config", agent_tool.config)
            agent_tool.save()

        serializer = AgentToolSerializer(agent_tool)
        return Response(serializer.data, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)

    @action(detail=True, methods=["post"])
    def remove_tool(self, request: Request, pk: int | None = None) -> Response:
        """Remove a tool from this agent.

        Request body:
            tool_id: int - ID of the tool to remove
        """
        agent = self.get_object()
        tool_id = request.data.get("tool_id")

        if not tool_id:
            return Response(
                {"error": "tool_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            agent_tool = AgentTool.objects.get(agent=agent, tool_id=tool_id)
            agent_tool.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except AgentTool.DoesNotExist:
            return Response(
                {"error": "Tool not assigned to this agent"},
                status=status.HTTP_404_NOT_FOUND,
            )

    @action(detail=True, methods=["post"])
    def memory_search(self, request: Request, pk: int | None = None) -> Response:
        """Search the agent's conversation memory.

        Request body:
            query: str - Search query
            search_type: str - "hybrid", "vector", or "text" (default: hybrid)
            max_results: int - Maximum results (1-10, default: 5)
            session_id: int - Optional session ID to limit search

        Returns:
            Response with search results
        """
        agent = self.get_object()

        if not agent.memory_search_enabled:
            return Response(
                {"error": "Memory search is not enabled for this agent"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer = MemorySearchRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data

        try:
            from memory.models import Session  # noqa: PLC0415
            from memory.search import MemorySearchConfig, MemorySearchService  # noqa: PLC0415

            # Get agent's memory search config
            config_data = agent.memory_search_config or {}
            config = MemorySearchConfig(
                max_results=data.get("max_results", config_data.get("max_results", 5)),
                min_score=config_data.get("min_score", 0.35),
                hybrid_weights=config_data.get("hybrid_weights", {"vector": 0.7, "text": 0.3}),
            )

            service = MemorySearchService(config)

            # Get session if provided
            session = None
            session_id = data.get("session_id")
            if session_id:
                try:
                    session = Session.objects.get(id=session_id, agent=agent)
                except Session.DoesNotExist:
                    return Response(
                        {"error": "Session not found"},
                        status=status.HTTP_404_NOT_FOUND,
                    )

            # Perform search
            results = service.search(
                query=data["query"],
                session=session,
                agent_id=agent.id,
                search_type=data.get("search_type", "hybrid"),
            )

            # Format response
            response_data = {
                "query": data["query"],
                "search_type": data.get("search_type", "hybrid"),
                "results": [
                    {
                        "content": r.content,
                        "score": round(r.score, 3),
                        "source": r.source,
                        "message_id": r.message_id,
                        "summary_id": r.summary_id,
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
                "count": len(results),
            }

            response_serializer = MemorySearchResponseSerializer(response_data)
            return Response(response_serializer.data)

        except Exception:
            logger.exception("Memory search error for agent %s", agent.name)
            return Response(
                {"error": "Failed to perform memory search"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ToolViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for listing available tools."""

    permission_classes = [IsAuthenticated]
    serializer_class = ToolSerializer
    queryset = Tool.objects.filter(is_active=True)
