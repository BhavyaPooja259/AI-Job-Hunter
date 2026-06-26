"""
Notification delivery layer.

Architecture
------------
NotificationChannel is an abstract base class.  Each concrete subclass
delivers a digest through one medium:

    ConsoleChannel    — print to stdout           (Sprint 14, implemented)
    EmailChannel      — SMTP / SendGrid            (future sprint)
    TelegramChannel   — Bot API                    (future sprint)
    SlackChannel      — Incoming Webhooks          (future sprint)
    DiscordChannel    — Webhook or Bot             (future sprint)

NotificationService holds an ordered list of channels and calls .send() on
each.  Failures are caught per-channel so a broken email gateway never
silences the console output.  The service returns two lists — channel names
that succeeded and channel names that failed — so the caller (NotificationAgent)
can build a typed NotificationResult without this module knowing anything about
the agent layer.

Adding a new channel in a future sprint
----------------------------------------
1. Create a class that inherits NotificationChannel.
2. Implement name (property) and send(digest) -> bool.
3. Pass an instance to NotificationService([..., NewChannel()]).
No changes to NotificationAgent or existing channels required.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from matching.digest import JobDigest

logger = logging.getLogger(__name__)

_BANNER = "=" * 62
_DIVIDER = "-" * 58


class NotificationChannel(ABC):
    """Abstract delivery channel.  Subclass this to add a new medium."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and NotificationResult."""

    @abstractmethod
    def send(self, digest: JobDigest) -> bool:
        """
        Deliver the digest through this channel.

        Returns True on success, False if the channel is available but the
        send failed for a non-exceptional reason (e.g. rate limit, empty
        recipient list).  Raise an exception for hard failures — the service
        will catch it and mark the channel as failed.
        """


class ConsoleChannel(NotificationChannel):
    """
    Prints the daily digest to stdout.

    Always returns True — console output cannot fail silently.
    Designed to be human-readable in a terminal with an 80-column width.
    """

    @property
    def name(self) -> str:
        return "console"

    def send(self, digest: JobDigest) -> bool:
        ts = digest.generated_at.strftime("%a %d %b %Y, %H:%M")
        dupes = digest.total_in_db - digest.unique_count

        lines: list[str] = [
            "",
            _BANNER,
            f"  Daily Job Digest — {ts}",
            (
                f"  {digest.total_in_db} in DB"
                + (f" ({dupes} duplicate{'s' if dupes != 1 else ''} removed)" if dupes else "")
                + f"  ·  {digest.unique_count} unique"
                + f"  ·  {len(digest.top_jobs)} top match{'es' if len(digest.top_jobs) != 1 else ''}"
                + f" (score ≥ {digest.score_threshold})"
            ),
            _BANNER,
            "",
        ]

        if digest.top_jobs:
            lines += [
                "  TOP MATCHES",
                "  " + _DIVIDER,
            ]
            for dj in digest.top_jobs:
                loc = dj.job.location or "Remote?"
                skills = ", ".join(dj.matched_skills[:3]) or "—"
                lines.append(
                    f"  [{dj.score:>3}]  {dj.job.title:<34} @ {dj.job.company:<18} {loc}"
                )
                lines.append(f"         Matched: {skills}")
            lines.append("")

        if digest.all_jobs:
            lines += [
                "  ALL JOBS  (sorted by match score)",
                "  " + _DIVIDER,
            ]
            for dj in digest.all_jobs:
                loc = dj.job.location or "Remote?"
                lines.append(
                    f"  [{dj.score:>3}]  {dj.job.title:<34} @ {dj.job.company:<18} {loc}"
                )
            lines.append("")

        by_company = digest.by_company
        if by_company:
            lines += [
                "  BY COMPANY",
                "  " + _DIVIDER,
            ]
            for company, jobs in sorted(by_company.items(), key=lambda kv: -max(dj.score for dj in kv[1])):
                lines.append(f"  {company}  ({len(jobs)} job{'s' if len(jobs) != 1 else ''})")
                for dj in jobs:
                    loc = dj.job.location or "Remote?"
                    lines.append(f"    [{dj.score:>3}]  {dj.job.title}  ({loc})")
            lines.append("")

        lines += [_BANNER, ""]

        print("\n".join(lines))
        return True


class NotificationService:
    """
    Routes a JobDigest to all registered channels.

    Usage
    -----
    from services.notification_service import NotificationService, ConsoleChannel

    service = NotificationService(channels=[ConsoleChannel()])
    notified, failed = service.send(digest)
    """

    def __init__(self, channels: list[NotificationChannel] | None = None) -> None:
        self._channels: list[NotificationChannel] = channels if channels is not None else []

    def add_channel(self, channel: NotificationChannel) -> None:
        """Register an additional channel at runtime."""
        self._channels.append(channel)

    @property
    def channel_names(self) -> list[str]:
        """Names of all registered channels."""
        return [c.name for c in self._channels]

    def send(self, digest: JobDigest) -> tuple[list[str], list[str]]:
        """
        Send the digest to all registered channels.

        Returns
        -------
        (notified, failed)
            notified — channel names that returned True
            failed   — channel names that returned False or raised
        """
        notified: list[str] = []
        failed: list[str] = []

        for channel in self._channels:
            try:
                ok = channel.send(digest)
                if ok:
                    notified.append(channel.name)
                    logger.info("digest sent via %s", channel.name)
                else:
                    failed.append(channel.name)
                    logger.warning("channel %s returned False", channel.name)
            except Exception as exc:
                failed.append(channel.name)
                logger.warning("channel %s raised: %s", channel.name, exc)

        return notified, failed
