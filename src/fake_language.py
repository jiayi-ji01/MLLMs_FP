"""Fake language rules for English -> Fake transfer experiments.

This module keeps the fake language intentionally simple and fully
deterministic. Relation names receive fixed aliases, and templates can be
rendered with placeholders such as ``{subject}`` and ``{relation_alias}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Mapping, Any


TOKEN_PATTERN = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")


def _invert_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    return {value: key for key, value in mapping.items()}


def _apply_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper() and source[1:].islower():
        return target.capitalize()
    return target


@dataclass(frozen=True)
class FakeLanguageConfig:
    """Configuration container for deterministic fake-language rendering."""

    relation_aliases: Mapping[str, str]
    token_map: Mapping[str, str] = field(default_factory=dict)
    fallback_char_map: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        fake_language_cfg: Mapping[str, Any] | None,
        relation_aliases: Mapping[str, str],
    ) -> "FakeLanguageConfig":
        fake_language_cfg = fake_language_cfg or {}
        return cls(
            relation_aliases=relation_aliases,
            token_map=fake_language_cfg.get("token_map", {}),
            fallback_char_map=fake_language_cfg.get("fallback_char_map", {}),
        )


class FakeLanguage:
    """Render fake-language prompts with fixed aliases and token substitutions."""

    def __init__(self, config: FakeLanguageConfig) -> None:
        self.config = config
        self._inverse_relation_aliases = _invert_mapping(config.relation_aliases)
        self._inverse_token_map = _invert_mapping(config.token_map)
        self._inverse_char_map = _invert_mapping(config.fallback_char_map)

    def relation_alias(self, relation_key: str) -> str:
        """Return the fake alias for a relation key."""
        if relation_key not in self.config.relation_aliases:
            raise KeyError(f"Unknown relation key: {relation_key}")
        return self.config.relation_aliases[relation_key]

    def render_question(self, template: str, relation_key: str, **kwargs: Any) -> str:
        """Render a fake template with relation aliases and placeholders."""
        render_kwargs = dict(kwargs)
        render_kwargs["relation_alias"] = self.relation_alias(relation_key)
        return template.format(**render_kwargs)

    def translate_token(self, token: str) -> str:
        """Translate one token while keeping placeholders and punctuation intact."""
        if not token or token.startswith("{") and token.endswith("}"):
            return token
        if token.isdigit() or re.fullmatch(r"[^\w\s]", token):
            return token

        lowered = token.lower()
        if lowered in self.config.token_map:
            return _apply_case(token, self.config.token_map[lowered])

        translated = "".join(self.config.fallback_char_map.get(char, char) for char in lowered)
        return _apply_case(token, translated)

    def invert_token(self, token: str) -> str:
        """Best-effort inverse for debugging generated fake prompts."""
        if not token or token.startswith("{") and token.endswith("}"):
            return token
        if token.isdigit() or re.fullmatch(r"[^\w\s]", token):
            return token

        lowered = token.lower()
        if lowered in self._inverse_relation_aliases:
            return _apply_case(token, self._inverse_relation_aliases[lowered])
        if lowered in self._inverse_token_map:
            return _apply_case(token, self._inverse_token_map[lowered])

        inverted = "".join(self._inverse_char_map.get(char, char) for char in lowered)
        return _apply_case(token, inverted)

    def translate_text(self, text: str) -> str:
        """Translate free English text into the fake language."""
        return self._transform_text(text=text, token_transform=self.translate_token)

    def invert_text(self, text: str) -> str:
        """Invert fake-language text back into approximate English."""
        return self._transform_text(text=text, token_transform=self.invert_token)

    def _transform_text(self, text: str, token_transform) -> str:
        parts: list[str] = []
        last_end = 0
        for match in TOKEN_PATTERN.finditer(text):
            start, end = match.span()
            parts.append(text[last_end:start])
            parts.append(token_transform(match.group()))
            last_end = end
        parts.append(text[last_end:])
        return "".join(parts)


__all__ = ["FakeLanguage", "FakeLanguageConfig"]
