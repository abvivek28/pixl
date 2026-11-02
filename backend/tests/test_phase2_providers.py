from io import BytesIO
import base64, json
import httpx
import numpy as np
from PIL import Image
import pytest
from app.providers.cloudflare_klein4b import CloudflareKlein4B, ProviderError, edit_prompt
from app.planning.meta_planner import MetaPlanner, MetaEditPlan, PlanningResult, convert_box_2d_to_pixels
from app.engine.deterministic import Box
from app.engine.orchestrator import run_edit, FakeHueShiftProvider

def png_bytes(color=(230, 30, 20), size=(32, 24)):
    buf=BytesIO(); Image.new("RGB", size, color).save(buf, format="PNG"); return buf.getvalue()

def test_edit_prompt_preserve_template():
    prompt=edit_prompt("make the car blue", ["keep the road", "keep lighting"])
    assert "Edit image 0: make the car blue" in prompt
    assert "keep the road; keep lighting" in prompt
    assert "Match the surrounding lighting" in prompt
    assert "Maintain the exact framing, composition, and aspect ratio" in prompt
    assert "Return only the edited image, at the exact same framing" in prompt
    assert "relative size, position, shape, and proportions" in prompt
    assert "Preserve the original background and overall colour grade" in prompt

def test_cloudflare_retries_429_and_caches(tmp_path):
    image_response=png_bytes((5, 220, 70))
    calls={"n":0}
    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1: return httpx.Response(429, headers={"retry-after":"0"}, json={"error":"rate limited"})
        assert request.headers.get("content-type", "").startswith("multipart/form-data")
        assert b"input_image_0" in request.content
        return httpx.Response(200, headers={"content-type":"image/png"}, content=image_response)
    client=httpx.Client(transport=httpx.MockTransport(handler))
    provider=CloudflareKlein4B("account", "token", cache_dir=tmp_path, client=client, retries=2, sleep=lambda _:None)
    input_image=Image.new("RGB", (32,24), (20,30,40))
    result=provider.edit(input_image, "change color", (256,256), 4, preserve=["everything else"])
    assert result.getpixel((0,0)) == (5,220,70)
    assert calls["n"] == 2
    # Same key comes from disk cache and must not make a third HTTP call.
    again=provider.edit(input_image, "change color", (256,256), 4, preserve=["everything else"])
    assert again.size == (32,24)
    assert calls["n"] == 2

def test_cloudflare_missing_credentials_is_actionable():
    provider=CloudflareKlein4B("", "", cache_dir="/tmp/pixl-test-cache")
    with pytest.raises(ProviderError, match="CLOUDFLARE_ACCOUNT_ID"):
        provider.edit(Image.new("RGB",(16,16)), "edit", (256,256), 0)

def test_meta_plan_structured_response_and_needs_confirmation():
    plan={"target_phrase":"the car","edit_type":"recolor","instruction":"make the car blue",
          "preserve":["road","lighting"],"box_2d":[100,100,700,800],"confidence":0.92,"questions":[],"match_count":1}
    response=httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(plan)}}]})
    client=httpx.Client(transport=httpx.MockTransport(lambda req: response))
    planner=MetaPlanner(api_key="test",client=client,min_interval_seconds=0)
    result=planner.plan(Image.new("RGB",(80,60)),"make the car blue")
    assert result.status == "planned"
    assert result.plan.target_phrase == "the car"
    assert result.coordinate_status == "unverified_raw_coordinates"

def test_meta_missing_key_falls_back_to_manual_box():
    result=MetaPlanner(api_key="").plan(Image.new("RGB",(20,20)),"remove the cup")
    assert result.status == "manual_fallback"
    assert "draw" in result.message or "box" in result.message

def test_ambiguous_meta_plan_requests_choice():
    plan={"target_phrase":"person","edit_type":"recolor","instruction":"change jacket","preserve":[],
          "box_2d":[100,100,800,800],"confidence":0.7,"questions":[{"question":"Which person?","options":["Left","Right"]}],"match_count":2}
    response=httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(plan)}}]})
    planner=MetaPlanner(api_key="test",client=httpx.Client(transport=httpx.MockTransport(lambda req: response)),min_interval_seconds=0)
    assert planner.plan(Image.new("RGB",(40,40)),"change jacket").status == "needs_choice"

def test_coordinate_conversion_requires_explicit_five_image_verification():
    with pytest.raises(ValueError, match="unverified"):
        convert_box_2d_to_pixels([100,200,500,700],1000,500)
    assert convert_box_2d_to_pixels([100,200,500,700],1000,500,
        verified_format="ymin_xmin_ymax_xmax_0_1000") == (200,50,700,250)

def test_run_edit_fake_provider_receipt_and_outside_invariant(tmp_path):
    source=tmp_path/"source.png"
    Image.fromarray(np.random.default_rng(4).integers(0,256,(80,100,3),dtype=np.uint8)).save(source)
    result=run_edit(source,"recolor selected area",tmp_path/"out",selected_box=Box(20,15,70,65),provider=FakeHueShiftProvider())
    assert result["metrics"]["outside_changed_pixels"] == 0
    assert (tmp_path/"out"/"result.png").exists()
    assert (tmp_path/"out"/"heatmap.png").exists()
    assert (tmp_path/"out"/"receipt.json").exists()

def test_global_edit_returns_warning_even_without_box():
    plan={"target_phrase":"whole image","edit_type":"global","instruction":"make the image look cinematic",
          "preserve":[],"box_2d":None,"confidence":0.9,"questions":[],"match_count":0}
    response=httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(plan)}}]})
    planner=MetaPlanner(api_key="test",client=httpx.Client(transport=httpx.MockTransport(lambda req: response)),min_interval_seconds=0)
    result=planner.plan(Image.new("RGB",(40,40)),"make the whole image cinematic")
    assert result.status == "global_warning"


def test_meta_default_model_is_current_flash_family():
    planner = MetaPlanner(api_key="test", min_interval_seconds=0)
    assert planner.model == "muse-spark-1.3-contributor"


def test_meta_request_uses_official_endpoint_model_auth_and_structured_json():
    plan={"target_phrase":"cup","edit_type":"recolor","instruction":"make the cup blue",
          "preserve":["table"],"box_2d":[100,100,700,700],"confidence":0.9,"questions":[],"match_count":1}
    captured={}
    def handler(request):
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        body=json.loads(request.content)
        captured["body"] = body
        return httpx.Response(200, json={"choices":[{"message":{"content":json.dumps(plan)}}]})
    planner=MetaPlanner(api_key="meta-test-key", client=httpx.Client(transport=httpx.MockTransport(handler)), min_interval_seconds=0)
    result=planner.plan(Image.new("RGB",(40,30)),"make the cup blue")
    assert result.status == "planned"
    assert captured["url"] == "https://api.meta.ai/v1/chat/completions"
    assert captured["auth"] == "Bearer meta-test-key"
    assert captured["body"]["model"] == "muse-spark-1.3-contributor"
    assert captured["body"]["response_format"]["type"] == "json_schema"
    assert captured["body"]["messages"][0]["content"][1]["type"] == "image_url"


def test_run_edit_uses_meta_instruction_preserve_and_selected_box(tmp_path):
    source = tmp_path / "plan-source.png"
    Image.new("RGB", (100, 80), (120, 120, 120)).save(source)
    captured = {}

    class RecordingProvider:
        name = "recording"
        model = "recording-test"
        def edit(self, image, instruction, out_size, seed, *, preserve=None):
            captured["instruction"] = instruction
            captured["preserve"] = preserve
            return Image.new("RGB", image.size, (200, 30, 20))

    result = run_edit(
        source, "original user prompt", tmp_path / "planned-out",
        selected_box=Box(20, 15, 70, 65), provider=RecordingProvider(),
        plan_data={"instruction": "make the selected object blue", "preserve": ["background", "lighting"]},
    )
    assert captured["instruction"] == "make the selected object blue"
    assert captured["preserve"] == ["background", "lighting"]
    assert result["boxes"]["selected"] == {"x1": 20, "y1": 15, "x2": 70, "y2": 65}
    assert result["boxes"]["expanded"] == {"x1": 5, "y1": 0, "x2": 85, "y2": 80}
    assert result["metrics"]["outside_changed_pixels"] == 0


def test_run_edit_corrects_provider_aspect_ratio_without_stretching(tmp_path):
    from app.providers.cloudflare_klein4b import EditProvider

    class WrongAspectProvider(EditProvider):
        name = "wrong-aspect-test"
        model = "wrong-aspect-test-v1"
        def edit(self, image, instruction, out_size, seed, *, preserve=None):
            # Deliberately return a wide image for the square context crop. The engine
            # should center-crop to square before resizing, rather than stretching.
            return Image.new("RGB", (160, 40), (180, 40, 20))

    source = tmp_path / "portrait.png"
    Image.new("RGB", (120, 180), (30, 60, 90)).save(source)
    out = tmp_path / "out"
    result = run_edit(source, "make selected region red", out,
                      selected_box=Box(45, 30, 75, 60), provider=WrongAspectProvider())
    receipt = json.loads((out / "receipt.json").read_text())
    assert result["metrics"]["outside_changed_pixels"] == 0
    assert receipt["model_input"]["aspect_ratio_corrected_without_stretching"] is True
    assert receipt["model_input"]["provider_output_width"] == 160
    assert receipt["model_input"]["provider_output_height"] == 40
    with Image.open(out / "result.png") as final_image:
        assert final_image.size == (120, 180)


def test_square_selection_uses_fixed_15px_context_and_edge_padding(tmp_path):
    source = tmp_path / "edge.png"
    Image.new("RGB", (80, 80), (100, 120, 140)).save(source)
    captured = {}

    class RecordingProvider:
        name = "recording"
        model = "recording-v1"
        def edit(self, image, instruction, out_size, seed, *, preserve=None):
            captured["input_size"] = image.size
            captured["out_size"] = out_size
            return image.copy()

    out = tmp_path / "edge-out"
    result = run_edit(source, "only edit the selected object", out,
                      selected_box=Box(0, 0, 20, 20), provider=RecordingProvider())
    receipt = json.loads((out / "receipt.json").read_text())
    assert captured["input_size"] == (50, 50)
    assert captured["out_size"][0] == captured["out_size"][1]
    assert result["boxes"]["selected"] == {"x1": 0, "y1": 0, "x2": 20, "y2": 20}
    assert result["boxes"]["expanded"] == {"x1": -15, "y1": -15, "x2": 35, "y2": 35}
    assert receipt["model_input"]["context_margin_px"] == 15
    assert receipt["model_input"]["context_padding_px"] == {"left": 15, "top": 15, "right": 0, "bottom": 0}
    assert result["metrics"]["outside_changed_pixels"] == 0


def test_rectangular_selection_allowed_and_preserved(tmp_path):
    source = tmp_path / "rect.png"
    Image.new("RGB", (100, 80), (40, 80, 120)).save(source)
    out = tmp_path / "rect-out"
    # Width = 60, Height = 40 (ratio 1.5, definitely not square)
    result = run_edit(source, "edit rectangle area", out,
                      selected_box=Box(10, 10, 70, 50), provider=FakeHueShiftProvider())
    assert result["metrics"]["outside_changed_pixels"] == 0
    assert result["boxes"]["selected"] == {"x1": 10, "y1": 10, "x2": 70, "y2": 50}
    with Image.open(out / "result.png") as final_image:
        assert final_image.size == (100, 80)
