"""
Flask Web Application for SatQuery AI Interactive Demo.

Provides REST APIs and serves the interactive satellite analyst dashboard:
- Smart task routing with automatic observation binding
- Visual multitemporal change detection with side-by-side viewer & overlay
- Land-cover multi-label predictions with radar/optical evidence
- Observable 5-step execution trace
"""

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from flask import Flask, jsonify, render_template, request, send_file

from satquery.orchestrator import SatQueryEngine
from satquery.taxonomy import TASK_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _encode_image_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 PNG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _create_change_overlay(
    t2_img: Image.Image,
    binary_mask: np.ndarray,
    bbox: Optional[List[int]] = None
) -> Image.Image:
    """Create RGB overlay with change pixels highlighted in red and bbox drawn."""
    t2_arr = np.array(t2_img.convert("RGB")).astype(np.float32)
    overlay = t2_arr.copy()

    # Red tint for changed pixels: 40% original + 60% red
    change_mask = binary_mask.astype(bool)
    overlay[change_mask, 0] = overlay[change_mask, 0] * 0.4 + 255 * 0.6
    overlay[change_mask, 1] = overlay[change_mask, 1] * 0.4
    overlay[change_mask, 2] = overlay[change_mask, 2] * 0.4

    result_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

    # Draw bounding box if present
    if bbox is not None and len(bbox) == 4:
        draw = ImageDraw.Draw(result_img)
        ymin, xmin, ymax, xmax = bbox
        # Draw 2-pixel cyan border
        for offset in range(2):
            draw.rectangle(
                [xmin - offset, ymin - offset, xmax + offset, ymax + offset],
                outline=(0, 255, 255)
            )

    return result_img


def create_app() -> Flask:
    """Factory creating and configuring the SatQuery AI Demo Flask application."""
    templates_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, template_folder=str(templates_dir), static_folder=str(static_dir))

    # Preload catalogs
    oscd_catalog_path = REPO_ROOT / "data" / "change_detection_catalog.parquet"
    oscd_df = pd.read_parquet(oscd_catalog_path) if oscd_catalog_path.exists() else pd.DataFrame()

    ben_meta_path = REPO_ROOT / "data" / "val_full_metadata.parquet"
    ben_df = pd.read_parquet(ben_meta_path) if ben_meta_path.exists() else pd.DataFrame()

    ben_features_path = REPO_ROOT / "initial_trails" / "X_val_full.npy"
    ben_features = np.load(ben_features_path) if ben_features_path.exists() else None

    ben_features_p2_path = REPO_ROOT / "data" / "X_val_phase2.npy"
    ben_features_p2 = np.load(ben_features_p2_path) if ben_features_p2_path.exists() else None

    # Cached engines for both policy modes
    engines = {
        "provisional": SatQueryEngine.create_default(routing_policy="provisional"),
        "clarify": SatQueryEngine.create_default(routing_policy="clarify")
    }

    # Preset queries for interactive chips
    preset_queries = [
        {
            "category": "Change Detection",
            "task": "CHANGE_DETECTION",
            "query": "What changed in this area between 2024 and 2026?",
            "recommended_mode": "auto"
        },
        {
            "category": "Change Detection",
            "task": "CHANGE_DETECTION",
            "query": "Detect land-cover modifications between these two acquisition dates.",
            "recommended_mode": "auto"
        },
        {
            "category": "Land Cover",
            "task": "LAND_COVER_CLASSIFICATION",
            "query": "What land-cover classes are present in this satellite image?",
            "recommended_mode": "auto"
        },
        {
            "category": "Land Cover",
            "task": "LAND_COVER_CLASSIFICATION",
            "query": "Classify all surface types detected in this scene.",
            "recommended_mode": "auto"
        },
        {
            "category": "Optical-SAR Fusion",
            "task": "OPTICAL_SAR_FUSION",
            "query": "Compare radar backscatter and optical reflectance for this scene.",
            "recommended_mode": "auto"
        },
        {
            "category": "Unsupported (VQA)",
            "task": "CHANGE_VQA",
            "query": "Did the lake surface area shrink between the two dates?",
            "recommended_mode": "auto"
        },
        {
            "category": "Unsupported (Grounding)",
            "task": "VISUAL_GROUNDING",
            "query": "Locate the reservoir and highlight its boundary with a mask.",
            "recommended_mode": "auto"
        }
    ]

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/catalog", methods=["GET"])
    def get_catalog():
        """Return available demo scenes and samples."""
        cd_scenes = []
        if not oscd_df.empty:
            for idx, row in oscd_df.iterrows():
                cd_scenes.append({
                    "id": int(idx),
                    "name": str(row["scene_name"]),
                    "split": str(row["split"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "change_pct": float(row["change_pct"]),
                    "total_pixels": int(row["total_pixels"])
                })

        lc_samples = []
        if not ben_df.empty and ben_features is not None:
            for idx in range(min(15, len(ben_df))):
                row = ben_df.iloc[idx]
                lc_samples.append({
                    "id": int(idx),
                    "patch_id": str(row["patch_id"]),
                    "ground_truth_labels": list(row["labels"])
                })

        return jsonify({
            "status": "success",
            "change_detection_scenes": cd_scenes,
            "land_cover_samples": lc_samples,
            "preset_queries": preset_queries,
            "total_cd_scenes": len(cd_scenes),
            "total_lc_samples": len(lc_samples)
        })

    @app.route("/api/query", methods=["POST"])
    def process_query_endpoint():
        """Process an analytical natural-language query end-to-end with smart observation binding."""
        payload = request.get_json() or {}
        query = payload.get("query", "").strip()
        policy = payload.get("policy", "provisional")
        mode = payload.get("mode", "auto")
        cd_idx = payload.get("cd_sample_idx")
        sample_idx = payload.get("sample_idx")
        synthetic_identical = payload.get("synthetic_identical", False)
        cd_method = payload.get("cd_method", "standard")

        if not query:
            return jsonify({"error": "Query string is required."}), 400

        engine = engines.get(policy, engines["provisional"])

        # 1. First predict task to enable smart auto-routing
        route = engine.router.predict(query)
        routed_task = route.task

        # 2. Determine effective modality
        # If mode is 'auto', automatically select modality matching the routed task
        if mode == "auto":
            if routed_task == "CHANGE_DETECTION":
                effective_mode = "change_detection"
            elif routed_task in ("LAND_COVER_CLASSIFICATION", "OPTICAL_SAR_FUSION"):
                effective_mode = "land_cover"
            else:
                effective_mode = "query_only"
        else:
            effective_mode = mode

        # 3. Resolve input data according to effective modality
        input_data = None
        visual_assets = {}
        scene_meta = None

        # Check for custom uploaded imagery
        custom_t1_b64 = payload.get("custom_t1")
        custom_t2_b64 = payload.get("custom_t2")
        is_custom = bool(custom_t1_b64)

        if is_custom:
            try:
                raw_t1 = custom_t1_b64.split(",", 1)[1] if "," in custom_t1_b64 else custom_t1_b64
                im1 = Image.open(io.BytesIO(base64.b64decode(raw_t1))).convert("RGB")
                
                if custom_t2_b64:
                    raw_t2 = custom_t2_b64.split(",", 1)[1] if "," in custom_t2_b64 else custom_t2_b64
                    im2 = Image.open(io.BytesIO(base64.b64decode(raw_t2))).convert("RGB")
                    if im2.size != im1.size:
                        im2 = im2.resize(im1.size, Image.Resampling.BILINEAR)
                else:
                    im2 = im1.copy()

                visual_assets["t1_image"] = _encode_image_base64(im1)
                visual_assets["t2_image"] = _encode_image_base64(im2)

                if effective_mode == "change_detection" or (effective_mode == "auto" and custom_t2_b64):
                    effective_mode = "change_detection"
                    input_data = {
                        "t1": im1,
                        "t2": im2,
                        "scene_name": "Custom Uploaded Scene",
                        "product_type": "rgb_8bit",
                        "method": cd_method
                    }
                    scene_meta = {
                        "scene_name": "Custom Uploaded Scene",
                        "split": "custom",
                        "dimensions": f"{im1.width} x {im1.height} px",
                        "method": cd_method
                    }
                else:
                    effective_mode = "land_cover"
                    arr = np.array(im1).astype(np.float32)
                    r = arr[:, :, 0]
                    g = arr[:, :, 1]
                    b = arr[:, :, 2]
                    nir = np.clip(1.5 * g - 0.5 * r + 20, 0, 255)
                    vv = (0.299 * r + 0.587 * g + 0.114 * b)
                    vh = vv * 0.5
                    bands = [b, g, r, nir, g * 0.9, g * 0.8, r * 0.9, nir * 0.9, r * 0.7, r * 0.5, vv, vh]
                    feats = []
                    for band in bands:
                        feats.extend([
                            float(np.nanmean(band)), float(np.nanstd(band)),
                            float(np.nanmin(band)), float(np.nanmax(band)),
                            float(np.nanpercentile(band, 25)), float(np.nanpercentile(band, 50)),
                            float(np.nanpercentile(band, 75))
                        ])
                    feats = np.array(feats, dtype=np.float32)
                    input_data = {
                        "features": feats,
                        "patch_id": "custom_uploaded_patch",
                        "ground_truth_labels": []
                    }
                    visual_assets["mode"] = "land_cover"
                    visual_assets["patch_id"] = "custom_uploaded_patch"
                    visual_assets["ground_truth_labels"] = []
                    visual_assets["active_model"] = "Custom Upload (84 Multi-Spectral Inferred Features)"
            except Exception as e:
                print(f"[API Warning] Failed to process custom image: {e}")
                is_custom = False

        if not is_custom and effective_mode == "change_detection":
            target_cd_idx = int(cd_idx) if cd_idx is not None else 0
            scene_name = f"sample_scene_{target_cd_idx}"
            split = "test"
            if not oscd_df.empty and 0 <= target_cd_idx < len(oscd_df):
                row = oscd_df.iloc[target_cd_idx]
                scene_name = str(row["scene_name"])
                split = str(row.get("split", "test"))
                t1_path = REPO_ROOT / row["t1_path"]
                t2_path = REPO_ROOT / row["t1_path"] if synthetic_identical else REPO_ROOT / row["t2_path"]
            else:
                t1_path = REPO_ROOT / "data" / "oscd" / "synthetic_t1.png"
                t2_path = REPO_ROOT / "data" / "oscd" / "synthetic_t2.png"

            if t1_path.exists() and t2_path.exists():
                t1_in, t2_in = str(t1_path), str(t2_path)
                try:
                    with Image.open(t1_path) as im1, Image.open(t2_path) as im2:
                        visual_assets["t1_image"] = _encode_image_base64(im1)
                        visual_assets["t2_image"] = _encode_image_base64(im2)
                except Exception as e:
                    print(f"[API Warning] Failed to generate thumbnails: {e}")
            else:
                rng = np.random.RandomState(42 + target_cd_idx)
                arr1 = (rng.uniform(50, 200, (256, 256, 3))).astype(np.uint8)
                arr2 = arr1.copy() if synthetic_identical else (arr1 + rng.uniform(-20, 20, (256, 256, 3))).clip(0, 255).astype(np.uint8)
                im1 = Image.fromarray(arr1)
                im2 = Image.fromarray(arr2)
                t1_in, t2_in = im1, im2
                visual_assets["t1_image"] = _encode_image_base64(im1)
                visual_assets["t2_image"] = _encode_image_base64(im2)

            input_data = {
                "t1": t1_in,
                "t2": t2_in,
                "scene_name": scene_name,
                "product_type": "oscd_rgb",
                "method": cd_method
            }
            scene_meta = {
                "scene_name": scene_name,
                "split": split,
                "synthetic_identical": synthetic_identical,
                "method": cd_method
            }

        elif effective_mode == "land_cover":
            target_sample_idx = int(sample_idx) if sample_idx is not None else 0
            model_ver = payload.get("model_version", "phase2")
            feats = None
            if model_ver == "phase2" and ben_features_p2 is not None and 0 <= target_sample_idx < len(ben_features_p2):
                feats = ben_features_p2[target_sample_idx]
                active_model_desc = "Phase 2 LightGBM (488 features, per-class thresholds)"
            elif ben_features is not None and 0 <= target_sample_idx < len(ben_features):
                feats = ben_features[target_sample_idx]
                active_model_desc = "Baseline Random Forest (84 global statistical features)"
            else:
                rng = np.random.RandomState(42 + target_sample_idx)
                n_feats = 488 if model_ver == "phase2" else 84
                feats = rng.randn(n_feats).astype(np.float32)
                active_model_desc = "Phase 2 LightGBM (488 features, synthetic demo sample)" if model_ver == "phase2" else "Baseline Random Forest (84 features, synthetic demo sample)"

            row = ben_df.iloc[target_sample_idx] if not ben_df.empty and target_sample_idx < len(ben_df) else {}
            input_data = {
                "features": feats,
                "patch_id": str(row.get("patch_id", f"sample_{target_sample_idx}")),
                "ground_truth_labels": list(row.get("labels", []))
            }
            visual_assets["mode"] = "land_cover"
            visual_assets["patch_id"] = str(row.get("patch_id", f"sample_{target_sample_idx}"))
            visual_assets["ground_truth_labels"] = list(row.get("labels", []))
            visual_assets["active_model"] = active_model_desc

        # 4. Execute query through SatQueryEngine
        response = engine.process_query(query, input_data=input_data)
        print(
            f"[Demo API] Query: \"{query}\" | Effective Mode: {effective_mode} | "
            f"Task: {response.selected_task or response.routed_task} | Status: {response.status} | "
            f"Confidence: {response.text_confidence*100:.1f}%",
            flush=True
        )

        # 5. Generate change overlay if change detection executed successfully
        if (
            effective_mode == "change_detection"
            and response.status in ("SUCCESS", "PROVISIONAL_SUCCESS")
            and response.routed_task == "CHANGE_DETECTION"
            and input_data is not None
            and "t1_image" in visual_assets
        ):
            try:
                meta = response.evidence.sensor_feature_summary or {}
                bbox = meta.get("largest_change_cluster_bbox_pixel")
                from satquery.specialists.change_detection import TemporalDifferenceBaseline
                temp_baseline = TemporalDifferenceBaseline()
                im1_obj = Image.open(input_data["t1"]) if isinstance(input_data["t1"], (str, Path)) else input_data["t1"]
                im2_obj = Image.open(input_data["t2"]) if isinstance(input_data["t2"], (str, Path)) else input_data["t2"]
                _, pred_mask, _ = temp_baseline.compute_change_map(
                    np.array(im1_obj).astype(np.float32),
                    np.array(im2_obj).astype(np.float32),
                    np.ones((im1_obj.size[1], im1_obj.size[0]), dtype=bool),
                    product_type="rgb_8bit" if is_custom else "oscd_rgb",
                    source_dtype="uint8",
                    method=cd_method
                )
                overlay_img = _create_change_overlay(im2_obj, pred_mask, bbox=bbox)
                visual_assets["change_overlay"] = _encode_image_base64(overlay_img)
                visual_assets["cd_method"] = cd_method
            except Exception as e:
                print(f"[API Warning] Failed to compute change overlay: {e}")

        resp_payload = {
            "response": response.to_dict(),
            "visuals": visual_assets
        }
        if scene_meta:
            resp_payload["scene_metadata"] = scene_meta

        return jsonify(resp_payload)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
