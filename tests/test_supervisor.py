from agents.supervisor import apply_rubric


def test_rubric_blocks_finish_without_citations():
    assert apply_rubric(
        citations_empty=True,
        grilled=False,
        step_count=1,
        proposed="FINISH",
        close_requested=False,
        last_error="",
    ) == "retriever"


def test_rubric_allows_finish_on_close_without_citations():
    assert apply_rubric(
        citations_empty=True,
        grilled=False,
        step_count=1,
        proposed="intake",
        close_requested=True,
        last_error="",
    ) == "FINISH"


def test_rubric_grills_before_finishing():
    assert apply_rubric(
        citations_empty=False,
        grilled=False,
        step_count=2,
        proposed="FINISH",
        close_requested=False,
        last_error="",
    ) == "intake"


def test_rubric_caps_steps():
    assert apply_rubric(
        citations_empty=False,
        grilled=True,
        step_count=8,
        proposed="retriever",
        close_requested=False,
        last_error="",
    ) == "FINISH"


def test_rubric_finishes_on_last_error():
    assert apply_rubric(
        citations_empty=False,
        grilled=False,
        step_count=1,
        proposed="retriever",
        close_requested=False,
        last_error="boom",
    ) == "FINISH"


def test_rubric_never_regrills_the_pm_in_the_same_turn():
    """Re-running the interrogator on an unchanged transcript loops the graph."""
    assert apply_rubric(
        citations_empty=False,
        grilled=True,
        step_count=3,
        proposed="intake",
        close_requested=False,
        last_error="",
    ) == "FINISH"


def test_rubric_never_reruns_retriever_once_citations_exist():
    assert apply_rubric(
        citations_empty=False,
        grilled=True,
        step_count=3,
        proposed="retriever",
        close_requested=False,
        last_error="",
    ) == "FINISH"
