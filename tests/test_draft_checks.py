"""Rule checks for Live store drafts: free, deterministic, and the same for every model."""
from app.evals.draft_checks import score_draft, sms_segments, summarize_scores

LINK = "https://thread-dev.myshopify.com/checkouts/1/recover?key=k1"
PRODUCTS = ["The Multi-location Snowboard"]


def test_the_real_fable_draft_fails_on_length_link_name_and_filler():
    fable = ("Hi sinister, this is the team at thread-dev. We noticed The Multi-location Snowboard is still in "
             "your checkout, so if you have any questions about it, just reply here and we'll be glad to help.")
    s = score_draft(fable, first_name="sinister", products=PRODUCTS, link=LINK)
    failed = {c.name for c in s.failed}
    assert failed == {"short", "link", "first name", "no filler"}
    assert s.body_chars > 120 and s.segments == 2 and s.score < 60


def test_a_draft_that_follows_the_rules_scores_100():
    good = ("Hi Sinister, your Multi-location Snowboard is still saved at Thread Snow Co. "
            f"Free returns within 30 days. {LINK}")
    s = score_draft(good, first_name="sinister", products=PRODUCTS, link=LINK)
    assert s.score == 100 and not s.failed and s.body_chars <= 120


def test_pressure_discount_emoji_and_invented_links_are_caught():
    bad = "Sinister! 20% off your Snowboard if you hurry 🎿 www.deal.example"
    failed = {c.name for c in score_draft(bad, first_name="Sinister", products=PRODUCTS, link=None).failed}
    assert failed == {"no pressure", "no discount", "no emoji", "no invented link"}


def test_failed_generations_are_not_scored_and_sms_segments_follow_the_charset():
    assert score_draft("(No draft: gemini busy.)", first_name="A", products=[], link=None) is None
    assert summarize_scores([None]) is None
    assert [sms_segments("a" * n) for n in (160, 161, 306, 307)] == [1, 2, 2, 3]
    assert sms_segments("é" * 160) == 1  # é is in the GSM-7 alphabet
    assert sms_segments("ł" * 71) == 2  # outside GSM-7: 70 per SMS, then 67
