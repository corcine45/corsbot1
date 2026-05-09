"""
Context Priority System — intelligent prompt trimming based on context importance.

Priority Levels:
  CRITICAL (weight=4): current topic, active emotion
  HIGH (weight=3):     current activity, active relationship  
  MEDIUM (weight=2):   recent memories
  LOW (weight=1):      old preferences, identity facts

The system trims prompts intelligently by:
1. Assigning priority weights to different context types
2. Scoring each context item based on priority + recency + relevance
3. Trimming lowest-priority items first when approaching token limits
4. Always preserving CRITICAL context unless severely constrained
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

log = logging.getLogger("corsbot.priority")


class Priority(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class ContextItem:
    """A single piece of context with priority and metadata."""
    content: str
    priority: Priority
    category: str  # e.g., "topic", "emotion", "activity", "memory", "preference"
    updated_at: float = field(default_factory=time.time)
    relevance_score: float = 1.0  # 0.0-1.0, set by caller based on query similarity
    token_estimate: int = 0  # estimated tokens, set during processing

    @property
    def weight(self) -> float:
        """Calculate final weight for sorting/trimming decisions."""
        # Priority weight * relevance * recency decay
        priority_weight = self.priority.value
        
        # Recency decay: items older than 30 min get slightly downweighted
        age_minutes = (time.time() - self.updated_at) / 60
        recency_factor = max(0.7, 1.0 - (age_minutes / 60))  # min 0.7, max 1.0
        
        return priority_weight * self.relevance_score * recency_factor

    def token_count(self) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        if self.token_estimate > 0:
            return self.token_estimate
        return len(self.content) // 4 + 1


class ContextPriorityManager:
    """
    Manages context prioritization and intelligent trimming.
    
    Usage:
        manager = ContextPriorityManager(max_tokens=1500)
        
        # Add context items
        manager.add("topic=valorant", Priority.CRITICAL, "topic")
        manager.add("emotion=excited", Priority.CRITICAL, "emotion")
        manager.add("activity=playing game", Priority.HIGH, "activity")
        manager.add("memory=likes pizza", Priority.MEDIUM, "memory")
        
        # Get trimmed context
        trimmed = manager.get_trimmed_context()
    """
    
    # Default token budgets for different context sections
    DEFAULT_MAX_TOKENS = 1500
    CRITICAL_MIN_TOKENS = 200  # always reserve space for critical context
    
    # Priority → approximate token allocation ratios
    PRIORITY_ALLOCATIONS = {
        Priority.CRITICAL: 0.40,  # 40% of budget
        Priority.HIGH: 0.30,      # 30% of budget
        Priority.MEDIUM: 0.20,    # 20% of budget
        Priority.LOW: 0.10,       # 10% of budget
    }
    
    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.max_tokens = max_tokens
        self._items: list[ContextItem] = []
    
    def add(self, content: str, priority: Priority, category: str, 
            updated_at: Optional[float] = None, relevance_score: float = 1.0) -> None:
        """Add a context item."""
        if not content or not content.strip():
            return
        
        item = ContextItem(
            content=content.strip(),
            priority=priority,
            category=category,
            updated_at=updated_at or time.time(),
            relevance_score=max(0.0, min(1.0, relevance_score)),
        )
        item.token_estimate = item.token_count()
        self._items.append(item)
    
    def clear(self) -> None:
        """Clear all context items."""
        self._items.clear()
    
    def _sort_by_weight(self) -> list[ContextItem]:
        """Sort items by weight (highest first)."""
        return sorted(self._items, key=lambda x: x.weight, reverse=True)
    
    def get_trimmed_context(self, max_tokens: Optional[int] = None) -> str:
        """
        Return context items trimmed to fit within token budget.
        Items are trimmed from lowest priority first.
        Returns a formatted string with priority sections.
        """
        if not self._items:
            return ""
        
        budget = max_tokens or self.max_tokens
        sorted_items = self._sort_by_weight()
        
        # Phase 1: Calculate per-priority budgets
        priority_budgets = {}
        for priority, ratio in self.PRIORITY_ALLOCATIONS.items():
            priority_budgets[priority] = int(budget * ratio)
        
        # Phase 2: Select items within each priority level
        selected: list[ContextItem] = []
        used_tokens: dict[Priority, int] = {p: 0 for p in Priority}
        
        for item in sorted_items:
            tokens = item.token_count()
            priority = item.priority
            
            # Check if item fits in its priority budget
            if used_tokens[priority] + tokens <= priority_budgets[priority]:
                selected.append(item)
                used_tokens[priority] += tokens
            else:
                # Check if we can borrow from lower priority budgets
                remaining_lower = sum(
                    priority_budgets[p] - used_tokens[p] 
                    for p in Priority if p.value < priority.value
                )
                if used_tokens[priority] + tokens <= priority_budgets[priority] + remaining_lower:
                    selected.append(item)
                    used_tokens[priority] += tokens
        
        # Phase 3: Format output with priority sections
        return self._format_context(selected)
    
    def _format_context(self, items: list[ContextItem]) -> str:
        """Format selected items into a structured context string."""
        if not items:
            return ""
        
        # Group by priority
        groups: dict[Priority, list[ContextItem]] = {p: [] for p in Priority}
        for item in items:
            groups[item.priority].append(item)
        
        sections = []
        
        # CRITICAL section
        if groups[Priority.CRITICAL]:
            lines = [item.content for item in groups[Priority.CRITICAL]]
            sections.append(f"[CRITICAL]\n" + "\n".join(lines))
        
        # HIGH section
        if groups[Priority.HIGH]:
            lines = [item.content for item in groups[Priority.HIGH]]
            sections.append(f"[HIGH]\n" + "\n".join(lines))
        
        # MEDIUM section
        if groups[Priority.MEDIUM]:
            lines = [item.content for item in groups[Priority.MEDIUM]]
            sections.append(f"[MEDIUM]\n" + "\n".join(lines))
        
        # LOW section
        if groups[Priority.LOW]:
            lines = [item.content for item in groups[Priority.LOW]]
            sections.append(f"[LOW]\n" + "\n".join(lines))
        
        return "\n\n".join(sections)
    
    def get_token_usage(self) -> dict:
        """Return token usage statistics."""
        total = sum(item.token_count() for item in self._items)
        by_priority = {}
        for p in Priority:
            by_priority[p.name] = sum(
                item.token_count() for item in self._items if item.priority == p
            )
        return {
            "total_tokens": total,
            "max_tokens": self.max_tokens,
            "by_priority": by_priority,
            "item_count": len(self._items),
        }


# ── Convenience functions for integration ───────────────────────────────────── #

def build_prioritized_context(
    topic: str = "",
    emotion: str = "",
    activity: str = "",
    relationship: str = "",
    recent_memories: list[str] = None,
    old_preferences: list[str] = None,
    max_tokens: int = ContextPriorityManager.DEFAULT_MAX_TOKENS,
    query: str = "",
) -> str:
    """
    Build a prioritized context string from individual components.
    
    This is the main entry point for integrating priority-based context
    into the AI prompt building pipeline.
    """
    manager = ContextPriorityManager(max_tokens=max_tokens)
    now = time.time()
    
    # CRITICAL: current topic
    if topic:
        manager.add(f"topic: {topic}", Priority.CRITICAL, "topic", now, relevance_score=1.0)
    
    # CRITICAL: active emotion
    if emotion:
        manager.add(f"emotion: {emotion}", Priority.CRITICAL, "emotion", now, relevance_score=1.0)
    
    # HIGH: current activity
    if activity:
        manager.add(f"activity: {activity}", Priority.HIGH, "activity", now, relevance_score=0.9)
    
    # HIGH: active relationship
    if relationship:
        manager.add(f"relationship: {relationship}", Priority.HIGH, "relationship", now, relevance_score=0.85)
    
    # MEDIUM: recent memories (with relevance scoring based on query)
    if recent_memories:
        for i, memory in enumerate(recent_memories):
            # First few memories get higher relevance
            relevance = max(0.5, 1.0 - (i * 0.1))
            manager.add(memory, Priority.MEDIUM, "memory", now, relevance_score=relevance)
    
    # LOW: old preferences
    if old_preferences:
        for i, pref in enumerate(old_preferences):
            # Older preferences get lower relevance
            relevance = max(0.3, 0.8 - (i * 0.15))
            manager.add(pref, Priority.LOW, "preference", now - (i * 3600), relevance_score=relevance)
    
    return manager.get_trimmed_context()


def trim_context_by_priority(
    context_blocks: dict[str, str],
    max_tokens: int = ContextPriorityManager.DEFAULT_MAX_TOKENS,
) -> str:
    """
    Trim a dictionary of context blocks based on priority.
    
    Args:
        context_blocks: Dict mapping priority level names to content strings.
                       Keys can be: "critical", "high", "medium", "low"
        max_tokens: Maximum token budget
    
    Returns:
        Formatted context string with lower-priority items trimmed first.
    """
    manager = ContextPriorityManager(max_tokens=max_tokens)
    now = time.time()
    
    priority_map = {
        "critical": Priority.CRITICAL,
        "high": Priority.HIGH,
        "medium": Priority.MEDIUM,
        "low": Priority.LOW,
    }
    
    for key, content in context_blocks.items():
        if not content or not content.strip():
            continue
        
        priority = priority_map.get(key.lower(), Priority.LOW)
        
        # Split multi-line content into individual items
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        for i, line in enumerate(lines):
            relevance = max(0.5, 1.0 - (i * 0.1))
            manager.add(line, priority, key, now, relevance_score=relevance)
    
    return manager.get_trimmed_context()


def estimate_context_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    if not text:
        return 0
    # Rough estimation: ~4 characters per token for English text
    return len(text) // 4 + 1