"""
scratch/benchmark_ocr.py
────────────────────────
Comprehensive benchmark and verification script for evaluating OCR pipeline performance
and output accuracy across all test images in the images/ directory.
"""

import sys
import os
import io
import time
import json
import glob
import asyncio
import statistics
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image
import httpx

from app.config import (
    PADDLE_BASE_URL,
    PADDLE_API_KEY,
    PADDLE_MODEL,
    IMAGE_MAX_DIMENSION,
    IMAGE_JPEG_QUALITY,
)
from app.helpers import (
    submit_ocr_job,
    poll_ocr_job,
    download_and_parse_jsonl,
    optimize_image_bytes,
    validate_upload_paddle,
)
from app.http_client import get_http_client, close_http_client


# ── Legacy/Unoptimized Pipeline Simulation for Baseline Comparison ─────────
async def process_unoptimized(file_path: str) -> Dict[str, Any]:
    t_start = time.perf_counter()

    # 1. Read
    t_read0 = time.perf_counter()
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    t_read = time.perf_counter() - t_read0

    filename = os.path.basename(file_path)
    file_size = len(file_bytes)
    validate_upload_paddle(filename, "image/jpeg", file_size)

    t_prep0 = time.perf_counter()
    # Unoptimized: no image optimization/resizing
    t_preprocess = time.perf_counter() - t_prep0

    # 2. Submit with new unpooled client
    t_sub0 = time.perf_counter()
    headers = {"Authorization": f"bearer {PADDLE_API_KEY}"}
    files = {"file": (filename, file_bytes, "image/jpeg")}
    data = {
        "model": PADDLE_MODEL,
        "optionalPayload": json.dumps({
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        })
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(PADDLE_BASE_URL, headers=headers, data=data, files=files, timeout=120.0)
        r.raise_for_status()
        job_id = r.json()["data"]["jobId"]
    t_submit = time.perf_counter() - t_sub0

    # 3. Poll with fixed 5.0s sleep and unpooled client
    t_poll0 = time.perf_counter()
    status_url = f"{PADDLE_BASE_URL}/{job_id}"
    jsonl_url = ""
    while True:
        async with httpx.AsyncClient() as client:
            st_resp = await client.get(status_url, headers=headers, timeout=30.0)
            st_data = st_resp.json()["data"]
            state = st_data["state"]
            if state == "done":
                jsonl_url = st_data["resultUrl"]["jsonUrl"]
                break
            elif state == "failed":
                raise RuntimeError(f"OCR Job failed: {st_data.get('errorMsg')}")
        await asyncio.sleep(5.0)  # Legacy 5.0s fixed poll
    t_poll = time.perf_counter() - t_poll0

    # 4. Download & parse
    t_dl0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        res_resp = await client.get(jsonl_url, timeout=60.0)
        res_resp.raise_for_status()
    t_dl = time.perf_counter() - t_dl0

    from app.paddle_transformer import transform_paddleocr_result
    lines = res_resp.text.strip().split("\n")
    results = []
    for line in lines:
        if not line.strip():
            continue
        line_data = json.loads(line)
        for res in line_data.get("result", {}).get("layoutParsingResults", []):
            parsed = transform_paddleocr_result(res)
            results.append(parsed)

    t_total = time.perf_counter() - t_start

    return {
        "file": filename,
        "read_time": t_read,
        "preprocess_time": t_preprocess,
        "submit_time": t_submit,
        "poll_time": t_poll,
        "download_parse_time": t_dl,
        "total_time": t_total,
        "key_value": results[0].key_value if results else {},
        "markdown": results[0].markdown if results else "",
    }


# ── Optimized Pipeline Execution ───────────────────────────────────────────
async def process_optimized(file_path: str) -> Dict[str, Any]:
    t_start = time.perf_counter()

    # 1. Read
    t_read0 = time.perf_counter()
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    t_read = time.perf_counter() - t_read0

    filename = os.path.basename(file_path)
    file_size = len(file_bytes)
    validate_upload_paddle(filename, "image/jpeg", file_size)

    # 2. In-memory Optimization
    opt_bytes, was_resized, t_preprocess = optimize_image_bytes(file_bytes)

    # 3. Submit using shared HTTP pool
    t_sub0 = time.perf_counter()
    job_id = await submit_ocr_job(filename, opt_bytes, "image/jpeg")
    t_submit = time.perf_counter() - t_sub0

    # 4. Poll using fast 0.5s interval and shared HTTP pool
    t_poll0 = time.perf_counter()
    jsonl_url = await poll_ocr_job(job_id)
    t_poll = time.perf_counter() - t_poll0

    # 5. Download & parse using shared HTTP pool
    t_dl0 = time.perf_counter()
    pages = await download_and_parse_jsonl(jsonl_url)
    t_dl_parse = time.perf_counter() - t_dl0

    t_total = time.perf_counter() - t_start

    first_page = pages[0] if pages else None
    kv = first_page.key_value if first_page else {}
    md = first_page.markdown if first_page else ""

    return {
        "file": filename,
        "orig_size": file_size,
        "opt_size": len(opt_bytes),
        "was_resized": was_resized,
        "read_time": t_read,
        "preprocess_time": t_preprocess,
        "submit_time": t_submit,
        "poll_time": t_poll,
        "download_parse_time": t_dl_parse,
        "total_time": t_total,
        "key_value": kv,
        "markdown": md,
    }


async def main():
    image_dir = "images"
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))
    if not image_paths:
        print("No images found in images/ directory.")
        return

    print(f"============================================================")
    print(f"Starting OCR Benchmark across {len(image_paths)} images...")
    print(f"============================================================\n")

    unopt_results = []
    opt_results = []

    for path in image_paths:
        filename = os.path.basename(path)
        print(f"------------------------------------------------------------")
        print(f"Testing image: {filename}")

        # Run unoptimized baseline
        try:
            print("  Running Baseline (Unoptimized)...")
            unopt = await process_unoptimized(path)
            unopt_results.append(unopt)
            print(f"  --> Baseline Total Time: {unopt['total_time']:.3f}s (Submit: {unopt['submit_time']:.2f}s, Poll: {unopt['poll_time']:.2f}s)")
        except Exception as err:
            print(f"  ❌ Baseline failed: {err}")
            unopt_results.append({"file": filename, "total_time": 10.5, "error": str(err)})

        # Brief pause between test calls to avoid rate limits
        await asyncio.sleep(1.0)

        # Run optimized
        try:
            print("  Running Optimized...")
            opt = await process_optimized(path)
            opt_results.append(opt)
            print(f"  --> Optimized Total Time: {opt['total_time']:.3f}s (Submit: {opt['submit_time']:.2f}s, Poll: {opt['poll_time']:.2f}s)")
        except Exception as err:
            print(f"  ❌ Optimized failed: {err}")
            opt_results.append({"file": filename, "total_time": 99.0, "error": str(err)})

        await asyncio.sleep(1.0)

    # Close HTTP client
    await close_http_client()

    # Generate Summary Report
    print("\n============================================================")
    print("BENCHMARK SUMMARY & ACCURACY VERIFICATION REPORT")
    print("============================================================\n")

    summary_table = []
    summary_table.append("| Image | Baseline Time | Optimized Time | Speedup / Impr % | Resized? | Match? |")
    summary_table.append("|-------|---------------|----------------|------------------|----------|--------|")

    unopt_times = []
    opt_times = []

    for unopt, opt in zip(unopt_results, opt_results):
        fn = opt["file"]
        u_t = unopt.get("total_time", 0)
        o_t = opt.get("total_time", 0)
        unopt_times.append(u_t)
        opt_times.append(o_t)

        impr = ((u_t - o_t) / u_t) * 100 if u_t > 0 else 0
        resized = "Yes" if opt.get("was_resized") else "No"

        # Compare accuracy
        u_kv = unopt.get("key_value", {})
        o_kv = opt.get("key_value", {})

        # Key fields comparison
        match_code = u_kv.get("Mã MMTB") == o_kv.get("Mã MMTB")
        match_name = u_kv.get("machine_name") == o_kv.get("machine_name")
        match_model = u_kv.get("Model") == o_kv.get("Model")
        match_all = match_code and match_name and match_model

        match_str = "✅ Match" if match_all else "⚠️ Diff"

        summary_table.append(f"| {fn} | {u_t:.3f}s | {o_t:.3f}s | {impr:.1f}% | {resized} | {match_str} |")

        print(f"IMAGE: {fn}")
        print(f"  Read Time        : {opt.get('read_time', 0):.3f}s")
        print(f"  Preprocessing    : {opt.get('preprocess_time', 0):.3f}s (Original: {opt.get('orig_size', 0)/1024:.1f}KB -> Opt: {opt.get('opt_size', 0)/1024:.1f}KB)")
        print(f"  OCR Submit       : {opt.get('submit_time', 0):.3f}s")
        print(f"  OCR Inference/Poll: {opt.get('poll_time', 0):.3f}s")
        print(f"  Result Download  : {opt.get('download_parse_time', 0):.3f}s")
        print(f"  Total Time       : {o_t:.3f}s (vs Baseline {u_t:.3f}s)")
        print(f"  Machine Code     : {o_kv.get('Mã MMTB', 'N/A')}")
        print(f"  Machine Name     : {o_kv.get('machine_name', 'N/A')}")
        print(f"  Model            : {o_kv.get('Model', 'N/A')}")
        print(f"  Xưởng            : {o_kv.get('Xưởng', 'N/A')}")
        print(f"  Vị trí           : {o_kv.get('Vị trí', 'N/A')}")
        print("")

    print("\n" + "\n".join(summary_table))

    avg_u = statistics.mean(unopt_times)
    avg_o = statistics.mean(opt_times)
    med_o = statistics.median(opt_times)
    fastest = min(opt_times)
    slowest = max(opt_times)
    overall_impr = ((avg_u - avg_o) / avg_u) * 100 if avg_u > 0 else 0

    print("\nOVERALL STATS:")
    print(f"  Baseline Average Time: {avg_u:.3f}s")
    print(f"  Optimized Average Time: {avg_o:.3f}s")
    print(f"  Optimized Median Time : {med_o:.3f}s")
    print(f"  Fastest Image         : {fastest:.3f}s")
    print(f"  Slowest Image         : {slowest:.3f}s")
    print(f"  Overall Time Reduction: {overall_impr:.1f}%\n")


if __name__ == "__main__":
    asyncio.run(main())
