from agents.supervisor import apply_rubric


def test_rubric_blocks_finish_without_citations():
    assert apply_rubric(
        citations_empty=True,
        questions_empty=True,
        step_count=1,
        proposed="FINISH",
        close_requested=False,
        last_error="",
    ) == "retriever"


def test_rubric_allows_finish_on_close_without_citations():
    assert apply_rubric(
        citations_empty=True,
        questions_empty=True,
        step_count=1,
        proposed="intake",
        close_requested=True,
        last_error="",
    ) == "FINISH"


def test_rubric_blocks_finish_without_questions_when_open():
    assert apply_rubric(
        citations_empty=False,
        questions_empty=True,
        step_count=2,
        proposed="FINISH",
        close_requested=False,
        last_error="",
    ) == "intake"


def test_rubric_caps_steps():
    assert apply_rubric(
        citations_empty=False,
        questions_empty=False,
        step_count=8,
        proposed="retriever",
        close_requested=False,
        last_error="",
    ) == "FINISH"


def test_rubric_finishes_on_last_error():
    assert apply_rubric(
        citations_empty=False,
        questions_empty=False,
        step_count=1,
        proposed="retriever",
        close_requested=False,
        last_error="boom",
    ) == "FINISH"
