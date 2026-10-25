import asyncio
from datetime import datetime, timezone, timedelta
import json
import logging
from typing import Any, Callable, TypeVar
import uuid

from pya_durable.models import (
    ApprovalDecision,
    ApprovalRequestRecord,
    ApprovalStatus,
    utc_now_iso,
)
from pya_durable.storage import DurableStorage

logger = logging.getLogger("pya_durable.approvals")

F = TypeVar("F", bound=Callable[..., Any])


class ApprovalRequiredError(Exception):
    def __init__(
        self,
        challenge_token: str,
        tool_name: str,
        args: dict[str, Any],
        prompt_message: str,
        expires_at: str | None = None,
    ) -> None:
        msg = (
            f"Tool '{tool_name}' requires human approval. "
            f"Challenge token: {challenge_token}. Prompt: {prompt_message}"
        )
        super().__init__(msg)
        self.message = msg
        self.challenge_token = challenge_token
        self.tool_name = tool_name
        self.tool_args = args
        self.prompt_message = prompt_message
        self.expires_at = expires_at

    def __str__(self) -> str:
        return self.message


class ApprovalRejectedError(Exception):
    def __init__(self, challenge_token: str, reason: str = "") -> None:
        msg = f"Tool approval rejected for challenge '{challenge_token}'."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)
        self.challenge_token = challenge_token
        self.reason = reason


class ApprovalExpiredError(Exception):
    def __init__(self, challenge_token: str) -> None:
        super().__init__(f"Approval challenge '{challenge_token}' has expired.")
        self.challenge_token = challenge_token


class InvalidApprovalDecisionError(ValueError):
    pass


def approval_gate(
    prompt: str | None = None,
    timeout_seconds: float | None = None,
    require_approval: bool = True,
) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        fn._requires_approval = require_approval  # type: ignore[attr-defined]
        fn._approval_prompt = prompt  # type: ignore[attr-defined]
        fn._approval_timeout = timeout_seconds  # type: ignore[attr-defined]
        return fn

    return decorator


class ApprovalManager:
    def __init__(self, storage: DurableStorage) -> None:
        self.storage = storage

    async def create_challenge(
        self,
        task_id: str,
        conversation_id: str,
        tool_name: str,
        args: dict[str, Any],
        prompt_message: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ApprovalRequestRecord:
        token = f"appr-{uuid.uuid4().hex[:16]}"
        prompt = prompt_message or f"Approve execution of tool '{tool_name}' with parameters {args}"
        now = datetime.now(timezone.utc)
        expires_at = (
            (now + timedelta(seconds=timeout_seconds)).isoformat()
            if timeout_seconds is not None
            else None
        )

        args_json = json.dumps(args)
        record = ApprovalRequestRecord(
            challenge_token=token,
            task_id=task_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            args_json=args_json,
            prompt_message=prompt,
            status=ApprovalStatus.PENDING,
            created_at=now.isoformat(),
            expires_at=expires_at,
        )

        await self.storage.insert_approval_request(record)
        logger.info(
            "Created approval challenge token %s for tool %s (task %s)",
            token,
            tool_name,
            task_id,
        )

        # Notify real-time stream hub and multiplayer hub if available
        try:
            from pya_durable.streaming import ApprovalRequestedEvent, stream_hub, multiplayer_hub

            event = ApprovalRequestedEvent(
                challenge_token=token,
                task_id=task_id,
                conversation_id=conversation_id,
                tool_name=tool_name,
                args=args,
                prompt_message=prompt,
                expires_at=expires_at,
            )
            stream_hub.publish_agent_event(event)
            multiplayer_hub.publish_event(event)
        except Exception:
            pass

        return record

    async def get(self, challenge_token: str) -> ApprovalRequestRecord | None:
        record = await self.storage.get_approval_request(challenge_token)
        if record is None:
            return None

        # Check expiration dynamically
        if record.status == ApprovalStatus.PENDING and record.expires_at is not None:
            now = datetime.now(timezone.utc)
            try:
                exp_dt = datetime.fromisoformat(record.expires_at)
                if now > exp_dt:
                    decision = ApprovalDecision(
                        status=ApprovalStatus.EXPIRED,
                        approver="system",
                        reason="Approval deadline expired",
                        resolved_at=utc_now_iso(),
                    )
                    record = await self.storage.resolve_approval_request(
                        challenge_token=challenge_token,
                        status=ApprovalStatus.EXPIRED,
                        decision_json=decision.model_dump_json(),
                        resolved_at=decision.resolved_at,
                    )
            except Exception:
                pass

        return record

    async def list_pending(
        self,
        conversation_id: str | None = None,
    ) -> list[ApprovalRequestRecord]:
        pending = await self.storage.list_pending_approvals(conversation_id)
        valid: list[ApprovalRequestRecord] = []
        now = datetime.now(timezone.utc)

        for rec in pending:
            if rec.expires_at is not None:
                try:
                    exp_dt = datetime.fromisoformat(rec.expires_at)
                    if now > exp_dt:
                        decision = ApprovalDecision(
                            status=ApprovalStatus.EXPIRED,
                            approver="system",
                            reason="Approval deadline expired",
                            resolved_at=utc_now_iso(),
                        )
                        await self.storage.resolve_approval_request(
                            challenge_token=rec.challenge_token,
                            status=ApprovalStatus.EXPIRED,
                            decision_json=decision.model_dump_json(),
                            resolved_at=decision.resolved_at,
                        )
                        continue
                except Exception:
                    pass
            valid.append(rec)
        return valid

    async def resolve(
        self,
        challenge_token: str,
        approved: bool,
        reason: str = "",
        override_args: dict[str, Any] | None = None,
        approver: str = "operator",
    ) -> ApprovalRequestRecord:
        record = await self.get(challenge_token)
        if record is None:
            raise ValueError(f"Approval challenge '{challenge_token}' not found.")

        if record.status != ApprovalStatus.PENDING:
            raise InvalidApprovalDecisionError(
                f"Approval challenge '{challenge_token}' is already resolved with status '{record.status}'."
            )

        status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        decision = ApprovalDecision(
            status=status,
            approver=approver,
            reason=reason,
            override_args_json=json.dumps(override_args) if override_args is not None else None,
            resolved_at=utc_now_iso(),
        )

        resolved_record = await self.storage.resolve_approval_request(
            challenge_token=challenge_token,
            status=status,
            decision_json=decision.model_dump_json(),
            resolved_at=decision.resolved_at,
        )
        if resolved_record is None:
            raise RuntimeError(f"Failed to resolve approval request '{challenge_token}'")

        logger.info(
            "Resolved approval challenge token %s with status %s by %s",
            challenge_token,
            status.value,
            approver,
        )

        # Notify real-time stream hub and multiplayer hub if available
        try:
            from pya_durable.streaming import ApprovalResolvedEvent, stream_hub, multiplayer_hub

            event = ApprovalResolvedEvent(
                challenge_token=challenge_token,
                task_id=resolved_record.task_id,
                conversation_id=resolved_record.conversation_id,
                status=status.value,
                approver=approver,
                reason=reason,
                override_args=override_args,
                resolved_at=decision.resolved_at,
            )
            stream_hub.publish_agent_event(event)
            multiplayer_hub.publish_event(event)
        except Exception:
            pass

        return resolved_record

    async def wait_for_resolution(
        self,
        challenge_token: str,
        poll_interval: float = 0.05,
        timeout: float | None = None,
    ) -> ApprovalRequestRecord:
        start_time = asyncio.get_event_loop().time()
        while True:
            rec = await self.get(challenge_token)
            if rec is not None and rec.status != ApprovalStatus.PENDING:
                return rec

            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    raise asyncio.TimeoutError(
                        f"Timed out waiting for approval challenge '{challenge_token}' after {timeout}s"
                    )

            await asyncio.sleep(poll_interval)
