from hypothesis import HealthCheck, given, settings, strategies as st

from tldw_Server_API.app.api.v1.endpoints.slides import _normalize_slides
from tldw_Server_API.app.api.v1.schemas.slides_schemas import Slide, SlideLayout


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=5, unique=True))
def test_normalize_slides_orders_contiguous(orders):
    slides = [
        Slide(order=order, layout=SlideLayout.CONTENT, title=f"Slide {order}", content="Body")
        for order in orders
    ]
    normalized = _normalize_slides(slides)
    assert [slide.order for slide in normalized] == list(range(len(slides)))
