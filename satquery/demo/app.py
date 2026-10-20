"""
Flask Web Application for SatQuery AI Interactive Demo.

Provides REST APIs and serves the interactive satellite analyst dashboard:
- Task routing with confidence gauges and policy arbitration
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

    # Red tint for changed pixels: 50% original + 50% red
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
            "recommended_mode": "change_detection"
        },
        {
            "category": "Change Detection",
            "task": "CHANGE_DETECTION",
            "query": "Detect land-cover modifications between these two acquisition dates.",
            "recommended_mode": "change_detection"
        },
        {
            "category": "Land Cover",
            "task": "LAND_COVER_CLASSIFICATION",
            "query": "What land-cover classes are present in this satellite image?",
            "recommended_mode": "land_cover"
        },
        {
            "category": "Land Cover",
            "task": "LAND_COVER_CLASSIFICATION",
            "query": "Classify all surface types detected in this scene.",
            "recommended_mode": "land_cover"
        },
        {
            "category": "Optical-SAR Fusion",
            "task": "OPTICAL_SAR_FUSION",
            "query": "Compare radar backscatter and optical reflectance for this scene.",
            "recommended_mode": "land_cover"
        },
        {
            "category": "Unsupported (VQA)",
            "task": "CHANGE_VQA",
            "query": "Did the lake surface area shrink between the two dates?",
            "recommended_mode": "query_only"
        },
        {
            "category": "Unsupported (Grounding)",
            "task": "VISUAL_GROUNDING",
            "query": "Locate the reservoir and highlight its boundary with a mask.",
            "recommended_mode": "query_only"
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
        """Process an analytical natural-language query end-to-end."""
        payload = request.get_json() or {}
        query = payload.get("query", "").strip()
        policy = payload.get("policy", "provisional")
        mode = payload.get("mode", "auto")
        cd_idx = payload.get("cd_sample_idx")
        sample_idx = payload.get("sample_idx")
        synthetic_identical = payload.get("synthetic_identical", False)

        if not query:
            return jsonify({"error": "Query string is required."}), 400

        engine = engines.get(policy, engines["provisional"])

        # Resolve input data according to mode
        input_data = None
        visual_assets = {}

        if mode == "change_detection" or (mode == "auto" and cd_idx is not None):
            if cd_idx is not None and 0 <= int(cd_idx) < len(oscd_df):
                row = oscd_df.iloc[int(cd_idx)]
                t1_path = REPO_ROOT / row["t1_path"]
                t2_path = REPO_ROOT / row["t1_path"] if synthetic_identical else REPO_ROOT / row["t2_path"]

                input_data = {
                    "t1": str(t1_path),
                    "t2": str(t2_path),
                    "scene_name": str(row["scene_name"])
                }

                # Generate visual base64 thumbnails
                try:
                    with Image.open(t1_path) as im1, Image.open(t2_path) as im2:
                        visual_assets["t1_image"] = _encode_image_base64(im1)
                        visual_assets["t2_image"] = _encode_image_base64(im2)

                        # Compute change mask for overlay visualization
                        cd_specialist = engine.registry.get_specialist(engine.router.predict(query).task)
                        # Predict
                        res = engine.process_query(query, input_data=input_data)
                        
                        # Generate overlay if specialist succeeded
                        if res.status in ("SUCCESS", "PROVISIONAL_SUCCESS") and res.evidence.sensor_feature_summary:
                            meta = res.evidence.sensor_feature_summary
                            bbox = meta.get("largest_change_cluster_bbox_pixel")
                            
                            # Load actual mask array computed by specialist
                            from satquery.specialists.change_detection import TemporalDifferenceBaseline
                            temp_baseline = TemporalDifferenceBaseline()
                            val = temp_baseline.supports_input(input_data)
                            _, pred_mask, _ = temp_baseline.compute_change_map(
                                np.array(im1).astype(np.float32),
                                np.array(im2).astype(np.float32),
                                np.ones((im1.size[1], im1.size[0]), dtype=bool)
                            )
                            overlay_img = _create_change_overlay(im2, pred_mask, bbox=bbox)
                            visual_assets["change_overlay"] = _encode_image_base64(overlay_img)
                            
                        return jsonify({
                            "response": res.to_dict(),
                            "visuals": visual_assets,
                            "scene_metadata": {
                                "scene_name": str(row["scene_name"]),
                                "split": str(row["split"]),
                                "synthetic_identical": synthetic_identical
                            }
                        })
                except Exception as e:
                    pass

        elif mode == "land_cover" or (mode == "auto" and sample_idx is not None):
            if sample_idx is not None and ben_features is not None and 0 <= int(sample_idx) < len(ben_features):
                idx = int(sample_idx)
                feats = ben_features[idx]
                row = ben_df.iloc[idx] if not ben_df.empty else {}
                input_data = {
                    "features": feats,
                    "patch_id": str(row.get("patch_id", "sample")),
                    "ground_truth_labels": list(row.get("labels", []))
                }
                visual_assets["mode"] = "land_cover"
                visual_assets["patch_id"] = str(row.get("patch_id", "sample"))

        # Standard execution for queries without specific image pair
        response = engine.process_query(query, input_data=input_data)

        return jsonify({
            "response": response.to_dict(),
            "visuals": visual_assets
        })

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
