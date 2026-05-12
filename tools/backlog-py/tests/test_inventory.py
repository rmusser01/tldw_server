from backlog_py.compat.inventory import load_builtin_inventory


def test_inventory_starts_with_agent_critical_commands():
    inventory = load_builtin_inventory()
    names = [item.name for item in inventory.items]

    assert names[:5] == [
        "cli:help",
        "cli:task-list-plain",
        "cli:task-view-plain",
        "cli:search-plain",
        "cli:board",
    ]


def test_inventory_classifies_browser_and_interactive_deferrals():
    inventory = load_builtin_inventory()
    by_name = {item.name: item for item in inventory.items}

    assert by_name["browser:kanban-drag-drop"].classification == "browser-deferred"
    assert by_name["cli:interactive-board"].classification == "interactive-deferred"
