# models.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class Asset:
    type: str
    uri: str
    name: str = ""
    ext: str = "unknown"

@dataclass
class Decorator:
    name: str
    value: Optional[str] = None
    fallback: Optional['Decorator'] = None

@dataclass
class LorebookEntry:
    keys: List[str]
    content: str
    enabled: bool
    insertion_order: int
    use_regex: bool = False
    constant: Optional[bool] = None
    selective: Optional[bool] = None
    secondary_keys: Optional[List[str]] = None
    position: Optional[str] = None
    decorators: List[Decorator] = field(default_factory=list)
    # Optional Fields
    name: Optional[str] = None
    priority: Optional[int] = None
    id: Optional[Union[int, str]] = None
    comment: Optional[str] = None

@dataclass
class Lorebook:
    name: Optional[str] = None
    description: Optional[str] = None
    scan_depth: Optional[int] = None
    token_budget: Optional[int] = None
    recursive_scanning: Optional[bool] = None
    extensions: Dict[str, Any] = field(default_factory=dict)
    entries: List[LorebookEntry] = field(default_factory=list)

@dataclass
class CharacterCardV3Data:
    name: str
    description: str
    tags: List[str]
    creator: str
    character_version: str
    mes_example: str
    extensions: Dict[str, Any]
    system_prompt: str
    post_history_instructions: str
    first_mes: str
    alternate_greetings: List[str]
    personality: str
    scenario: str
    creator_notes: str
    character_book: Optional[Lorebook] = None
    assets: List[Asset] = field(default_factory=list)
    nickname: Optional[str] = None
    creator_notes_multilingual: Optional[Dict[str, str]] = None
    source: Optional[List[str]] = None
    group_only_greetings: List[str] = field(default_factory=list)
    creation_date: Optional[int] = None
    modification_date: Optional[int] = None

@dataclass
class CharacterCardV3:
    spec: str
    spec_version: str
    data: CharacterCardV3Data