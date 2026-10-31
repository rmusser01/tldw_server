import pytest

from tldw_Server_API.app.core.Chat.conversation_enrichment import (
    CLUSTER_ID_OPT_OUT,
    CLUSTER_ID_UNCLUSTERED,
    auto_tag_conversation,
    cluster_conversations_for_user,
)


def _create_character(db):
    return db.add_character_card(
        {
            "name": "Tagging Character",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": db.client_id,
        }
    )


@pytest.mark.unit
def test_auto_tag_idempotency_and_manual_override(chacha_db):
    character_id = _create_character(chacha_db)
    conversation_id = chacha_db.add_conversation(
        {
            "character_id": character_id,
            "title": "Payment Issue",
        }
    )

    for content in ("Billing broke yesterday", "Chargeback question", "Need invoice update"):
        chacha_db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": content,
            }
        )

    result = auto_tag_conversation(chacha_db, conversation_id, owner_user_id="test-user")
    assert result.updated is True

    conversation = chacha_db.get_conversation_by_id(conversation_id)
    assert conversation["topic_label_source"] == "auto"
    first_tagged_message_id = conversation.get("topic_last_tagged_message_id")
    first_version = conversation.get("version")

    result_repeat = auto_tag_conversation(chacha_db, conversation_id, owner_user_id="test-user")
    assert result_repeat.updated is False

    conversation_repeat = chacha_db.get_conversation_by_id(conversation_id)
    assert conversation_repeat.get("topic_last_tagged_message_id") == first_tagged_message_id
    assert conversation_repeat.get("version") == first_version

    for content in ("Follow up", "Receipt missing", "Thanks"):
        chacha_db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": content,
            }
        )

    manual_version = conversation_repeat.get("version")
    chacha_db.update_conversation(
        conversation_id,
        {"topic_label": "Manual Topic", "topic_label_source": "manual"},
        manual_version,
    )

    result_manual = auto_tag_conversation(chacha_db, conversation_id, owner_user_id="test-user")
    assert result_manual.updated is False
    conversation_manual = chacha_db.get_conversation_by_id(conversation_id)
    assert conversation_manual.get("topic_label") == "Manual Topic"
    assert conversation_manual.get("topic_label_source") == "manual"


@pytest.mark.unit
def test_clustering_persists_metadata(chacha_db):
    character_id = _create_character(chacha_db)
    conv_a = chacha_db.add_conversation(
        {"character_id": character_id, "title": "A", "topic_label": "Billing Issues"}
    )
    conv_b = chacha_db.add_conversation(
        {"character_id": character_id, "title": "B", "topic_label": "Billing Issues"}
    )

    result = cluster_conversations_for_user(chacha_db)
    assert result.clusters_written >= 1

    conv_a_row = chacha_db.get_conversation_by_id(conv_a)
    conv_b_row = chacha_db.get_conversation_by_id(conv_b)
    assert conv_a_row.get("cluster_id") == conv_b_row.get("cluster_id")

    cluster = chacha_db.get_conversation_cluster(conv_a_row.get("cluster_id"))
    assert cluster is not None
    assert cluster.get("size") == 2
    assert cluster.get("title") == "Billing Issues"


@pytest.mark.unit
def test_clustering_opt_out_and_unclustered(chacha_db):
    character_id = _create_character(chacha_db)
    conv_unclustered = chacha_db.add_conversation(
        {"character_id": character_id, "title": "No Topic"}
    )
    conv_opt_out = chacha_db.add_conversation(
        {
            "character_id": character_id,
            "title": "Opt Out",
            "topic_label": "Special",
            "cluster_id": CLUSTER_ID_OPT_OUT,
        }
    )

    cluster_conversations_for_user(chacha_db)

    unclustered_row = chacha_db.get_conversation_by_id(conv_unclustered)
    opt_out_row = chacha_db.get_conversation_by_id(conv_opt_out)

    assert unclustered_row.get("cluster_id") == CLUSTER_ID_UNCLUSTERED
    assert opt_out_row.get("cluster_id") == CLUSTER_ID_OPT_OUT

    cluster = chacha_db.get_conversation_cluster(CLUSTER_ID_UNCLUSTERED)
    assert cluster is not None
    assert cluster.get("size") == 1
