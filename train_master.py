"""
train_master.py
---------------
Script điều phối tổng - Chạy 1 lần để train GLiNER và/hoặc Level Classifier
tuỳ theo config trong config.yaml.

Sử dụng:
    python train_master.py                     # Dùng config.yaml mặc định
    python train_master.py --config my.yaml    # Dùng config khác
    python train_master.py --gliner-only       # Chỉ train GLiNER
    python train_master.py --classifier-only   # Chỉ train Classifier
    python train_master.py --check-deps        # Kiểm tra thư viện trước khi train

Luồng xử lý:
    1. Load config.yaml
    2. Kiểm tra các flag run.train_gliner / run.train_classifier
    3. Gọi train_gliner() và/hoặc train_classifier()
    4. In tóm tắt kết quả
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Thêm thư mục hiện tại vào sys.path
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config, set_seed, print_banner, format_time


# ----------------------------------------------------------------
# Kiểm tra dependencies
# ----------------------------------------------------------------
def check_dependencies(run_gliner: bool, run_classifier: bool) -> bool:
    """
    Kiểm tra tất cả các thư viện cần thiết đã được cài chưa.
    In ra hướng dẫn cài nếu thiếu.

    Args:
        run_gliner     : Có cần check GLiNER deps không
        run_classifier : Có cần check Classifier deps không

    Returns:
        True nếu tất cả OK, False nếu có thư viện thiếu
    """
    all_ok = True
    
    print("[Deps] Kiểm tra thư viện...")
    
    # Thư viện chung
    common_deps = {
        "torch": "pip install torch --index-url https://download.pytorch.org/whl/cu118",
        "transformers": "pip install transformers",
        "yaml": "pip install pyyaml",
        "numpy": "pip install numpy",
        "sklearn": "pip install scikit-learn",
        "tqdm": "pip install tqdm",
    }
    
    # Thư viện riêng cho GLiNER
    gliner_deps = {
        "gliner": "pip install gliner",
    }
    
    # Thư viện riêng cho Classifier
    classifier_deps = {
        "accelerate": "pip install accelerate",
    }
    
    deps_to_check = dict(common_deps)
    if run_gliner:
        deps_to_check.update(gliner_deps)
    if run_classifier:
        deps_to_check.update(classifier_deps)
    
    for pkg, install_cmd in deps_to_check.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} - CHƯA CÀI! Chạy: {install_cmd}")
            all_ok = False
    
    if all_ok:
        print("\n[Deps] Tất cả thư viện đã sẵn sàng!\n")
    else:
        print("\n[Deps] Có thư viện bị thiếu. Hãy cài đặt trước khi train.\n")
    
    return all_ok


# ----------------------------------------------------------------
# In tóm tắt config
# ----------------------------------------------------------------
def print_config_summary(cfg: dict):
    """In tóm tắt config để người dùng xác nhận."""
    print_banner("CẤU HÌNH TRAINING")
    
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    
    print(f"Dataset    : {data_cfg.get('dataset_path', 'N/A')}")
    print(f"Val ratio  : {data_cfg.get('val_ratio', 0.1)}")
    print(f"Max samples: {data_cfg.get('max_samples', 'Toàn bộ')}")
    print(f"Seed       : {run_cfg.get('seed', 42)}")
    print()
    
    train_gliner = run_cfg.get("train_gliner", False)
    train_classifier = run_cfg.get("train_classifier", False)
    
    if train_gliner:
        gcfg = cfg.get("gliner", {})
        print("GLiNER:")
        print(f"  Model       : {gcfg.get('model_name')}")
        print(f"  Entity types: {gcfg.get('entity_types')}")
        print(f"  Epochs      : {gcfg.get('num_epochs')}")
        print(f"  Batch size  : {gcfg.get('train_batch_size')}")
        print(f"  LR          : {gcfg.get('learning_rate')}")
        print(f"  Output      : {gcfg.get('output_dir')}")
        print()
    
    if train_classifier:
        ccfg = cfg.get("classifier", {})
        print("Level Classifier:")
        print(f"  Model      : {ccfg.get('model_name')}")
        print(f"  Levels     : {ccfg.get('level_labels')}")
        print(f"  Epochs     : {ccfg.get('num_epochs')}")
        print(f"  Batch size : {ccfg.get('train_batch_size')}")
        print(f"  LR         : {ccfg.get('learning_rate')}")
        print(f"  Truncation : {ccfg.get('truncation_strategy')}")
        print(f"  Output     : {ccfg.get('output_dir')}")
        print()


# ----------------------------------------------------------------
# Resolve đường dẫn tương đối của dataset
# ----------------------------------------------------------------
def resolve_dataset_path(cfg: dict, config_path: str) -> dict:
    """
    Nếu dataset_path là đường dẫn tương đối, resolve nó so với
    vị trí của config.yaml (không phải thư mục hiện tại).

    Args:
        cfg        : Config dict
        config_path: Đường dẫn tới file config.yaml

    Returns:
        Config dict với dataset_path đã resolve
    """
    dataset_path = cfg["data"]["dataset_path"]
    
    if not os.path.isabs(dataset_path):
        config_dir = Path(config_path).parent
        resolved = (config_dir / dataset_path).resolve()
        cfg["data"]["dataset_path"] = str(resolved)
        print(f"[Config] Dataset path: {cfg['data']['dataset_path']}")
    
    if not os.path.exists(cfg["data"]["dataset_path"]):
        print(f"\n[LỖI] Không tìm thấy dataset tại: {cfg['data']['dataset_path']}")
        print("Hãy kiểm tra lại 'data.dataset_path' trong config.yaml")
        sys.exit(1)
    
    return cfg


# ----------------------------------------------------------------
# Resolve đường dẫn output
# ----------------------------------------------------------------
def resolve_output_dirs(cfg: dict, config_path: str) -> dict:
    """
    Resolve đường dẫn output_dir tương đối so với config.yaml.
    
    Args:
        cfg        : Config dict
        config_path: Đường dẫn tới file config.yaml
    
    Returns:
        Config dict với output_dir đã resolve
    """
    config_dir = Path(config_path).parent
    
    for section in ["gliner", "classifier"]:
        if section in cfg and "output_dir" in cfg[section]:
            out = cfg[section]["output_dir"]
            if not os.path.isabs(out):
                resolved = (config_dir / out).resolve()
                cfg[section]["output_dir"] = str(resolved)
    
    return cfg


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    """
    Entry point chính của train_master.py.
    
    Parse arguments, load config, kiểm tra deps, rồi điều phối
    việc train GLiNER và/hoặc Level Classifier.
    """
    parser = argparse.ArgumentParser(
        description="Master Training Script - GLiNER NER + Level Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python train_master.py                          # Train theo config.yaml
  python train_master.py --config my_config.yaml  # Dùng config khác
  python train_master.py --gliner-only            # Chỉ train GLiNER
  python train_master.py --classifier-only        # Chỉ train Level Classifier  
  python train_master.py --check-deps             # Chỉ kiểm tra thư viện
        """
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Đường dẫn tới file config.yaml (mặc định: config.yaml)"
    )
    parser.add_argument(
        "--gliner-only", action="store_true",
        help="Chỉ train GLiNER, bỏ qua Classifier dù config thế nào"
    )
    parser.add_argument(
        "--classifier-only", action="store_true",
        help="Chỉ train Level Classifier, bỏ qua GLiNER"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Chạy benchmark so sánh nhiều model Classifier (bỏ qua GLiNER và Classifier đơn)"
    )
    parser.add_argument(
        "--benchmark-gliner", action="store_true", dest="benchmark_gliner",
        help="Chạy benchmark so sánh nhiều GLiNER architecture (SKILL + EXPERIENCE NER)"
    )
    parser.add_argument(
        "--check-deps", action="store_true",
        help="Chỉ kiểm tra thư viện, không train"
    )
    parser.add_argument(
        "--no-test", action="store_true",
        help="Bỏ qua bước quick test sau khi train"
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Danh sách các model GLiNER muốn chạy, cách nhau bằng dấu phẩy (vd: GLiNER-Small-v2.5,GLiNER-Medium-v2.5)"
    )
    args = parser.parse_args()
    
    # --- Resolve config path ---
    config_path = str(Path(args.config).resolve())
    if not os.path.exists(config_path):
        # Thử tìm trong cùng thư mục với script
        alt_path = str(Path(__file__).parent / args.config)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            print(f"[LỖI] Không tìm thấy config: {config_path}")
            sys.exit(1)
    
    print(f"[Config] Đang load: {config_path}")
    cfg = load_config(config_path)
    
    # --- Override từ command line args ---
    if args.models:
        target_models = [m.strip().lower() for m in args.models.split(",")]
        # Override cho benchmark_gliner
        if "benchmark_gliner" in cfg and "models" in cfg["benchmark_gliner"]:
            for m in cfg["benchmark_gliner"]["models"]:
                m_name = m.get("name", m["model_name"]).strip().lower()
                m_hf_id = m["model_name"].strip().lower()
                matched = False
                for tm in target_models:
                    if tm in m_name or tm in m_hf_id:
                        matched = True
                        break
                m["enabled"] = matched
        
        # Override cho single train gliner
        if "gliner" in cfg:
            first_target = target_models[0]
            matched_model_cfg = None
            if "benchmark_gliner" in cfg and "models" in cfg["benchmark_gliner"]:
                for m in cfg["benchmark_gliner"]["models"]:
                    m_name = m.get("name", m["model_name"]).strip().lower()
                    m_hf_id = m["model_name"].strip().lower()
                    if first_target in m_name or first_target in m_hf_id:
                        matched_model_cfg = m
                        break
            if matched_model_cfg:
                cfg["gliner"]["model_name"] = matched_model_cfg["model_name"]
                if "train_batch_size" in matched_model_cfg:
                    cfg["gliner"]["train_batch_size"] = matched_model_cfg["train_batch_size"]
                if "gradient_accumulation_steps" in matched_model_cfg:
                    cfg["gliner"]["gradient_accumulation_steps"] = matched_model_cfg["gradient_accumulation_steps"]
                if "learning_rate" in matched_model_cfg:
                    cfg["gliner"]["learning_rate"] = matched_model_cfg["learning_rate"]
            else:
                cfg["gliner"]["model_name"] = args.models
            print(f"[Config] Override model GLiNER đơn: {cfg['gliner']['model_name']}")
            print(f"         Hyperparams: batch={cfg['gliner'].get('train_batch_size')}, grad_accum={cfg['gliner'].get('gradient_accumulation_steps')}, lr={cfg['gliner'].get('learning_rate')}")

    if args.gliner_only:
        cfg["run"]["train_gliner"] = True
        cfg["run"]["train_classifier"] = False
        cfg["run"]["run_benchmark"] = False
        print("[Config] Override: Chỉ train GLiNER")

    if args.classifier_only:
        cfg["run"]["train_gliner"] = False
        cfg["run"]["train_classifier"] = True
        cfg["run"]["run_benchmark"] = False
        print("[Config] Override: Chỉ train Level Classifier")

    if args.benchmark:
        cfg["run"]["train_gliner"] = False
        cfg["run"]["train_classifier"] = False
        cfg["run"]["run_benchmark"] = True
        cfg["run"]["run_benchmark_gliner"] = False
        print("[Config] Override: Chạy Benchmark Classifier")

    if args.benchmark_gliner:
        cfg["run"]["train_gliner"] = False
        cfg["run"]["train_classifier"] = False
        cfg["run"]["run_benchmark"] = False
        cfg["run"]["run_benchmark_gliner"] = True
        print("[Config] Override: Chạy Benchmark GLiNER")

    run_gliner = cfg["run"].get("train_gliner", False)
    run_classifier = cfg["run"].get("train_classifier", False)
    run_benchmark = cfg["run"].get("run_benchmark", False)
    run_benchmark_gliner = cfg["run"].get("run_benchmark_gliner", False)

    if not run_gliner and not run_classifier and not run_benchmark and not run_benchmark_gliner:
        print("[LỖI] Không có tác vụ nào được bật trong config!")
        print("Hãy bật train_gliner / train_classifier / run_benchmark / run_benchmark_gliner.")
        sys.exit(1)
    
    # --- Kiểm tra deps ---
    deps_ok = check_dependencies(run_gliner or run_benchmark_gliner, run_classifier or run_benchmark)

    if args.check_deps:
        sys.exit(0 if deps_ok else 1)

    if not deps_ok:
        print("[LỖI] Có thư viện chưa cài. Hãy cài đặt rồi chạy lại.")
        sys.exit(1)
    
    # --- Resolve paths ---
    cfg = resolve_dataset_path(cfg, config_path)
    cfg = resolve_output_dirs(cfg, config_path)
    
    # --- In tóm tắt config ---
    print_config_summary(cfg)
    
    # --- Bắt đầu train ---
    overall_start = time.time()
    results = {}
    
    # ============================================================
    # TRAIN GLiNER
    # ============================================================
    if run_gliner:
        print_banner("BẮT ĐẦU TRAIN GLiNER")
        try:
            from train_gliner import train_gliner, quick_test_gliner
            
            gliner_start = time.time()
            gliner_output = train_gliner(cfg)
            gliner_elapsed = format_time(time.time() - gliner_start)
            
            results["gliner"] = {
                "status": "SUCCESS",
                "output": gliner_output,
                "time": gliner_elapsed,
            }
            
            # Quick test
            if not args.no_test:
                entity_types = cfg["gliner"].get("entity_types", ["SKILL", "MAJOR", "EXPERIENCE"])
                quick_test_gliner(gliner_output, entity_types)
        
        except KeyboardInterrupt:
            print("\n[GLiNER] Đã dừng bởi người dùng (Ctrl+C)")
            results["gliner"] = {"status": "INTERRUPTED"}
        except Exception as e:
            print(f"\n[GLiNER] LỖI: {e}")
            import traceback
            traceback.print_exc()
            results["gliner"] = {"status": "FAILED", "error": str(e)}
    
    # ============================================================
    # TRAIN CLASSIFIER
    # ============================================================
    if run_classifier:
        print_banner("BẮT ĐẦU TRAIN LEVEL CLASSIFIER")
        try:
            from train_classifier import train_classifier, quick_test_classifier
            
            clf_start = time.time()
            clf_output = train_classifier(cfg)
            clf_elapsed = format_time(time.time() - clf_start)
            
            results["classifier"] = {
                "status": "SUCCESS",
                "output": clf_output,
                "time": clf_elapsed,
            }
            
            # Quick test
            if not args.no_test:
                level_labels = cfg["classifier"].get("level_labels", [
                    "INTERN", "FRESHER", "JUNIOR", "MIDDLE",
                    "SENIOR", "LEAD", "MANAGER", "DIRECTOR", "EXPERT", "UNKNOWN"
                ])
                quick_test_classifier(clf_output, level_labels)
        
        except KeyboardInterrupt:
            print("\n[Classifier] Đã dừng bởi người dùng (Ctrl+C)")
            results["classifier"] = {"status": "INTERRUPTED"}
        except Exception as e:
            print(f"\n[Classifier] LỖI: {e}")
            import traceback
            traceback.print_exc()
            results["classifier"] = {"status": "FAILED", "error": str(e)}
    
    # ============================================================
    # BENCHMARK GLiNER
    # ============================================================
    if run_benchmark_gliner:
        print_banner("BẮT ĐẦU BENCHMARK GLiNER")
        try:
            from benchmark_gliner import run_all_gliner_benchmarks

            # Resolve output_dir cho benchmark_gliner
            bench_gliner_out = cfg.get("benchmark_gliner", {}).get(
                "output_dir", "./outputs/benchmark_gliner"
            )
            if not os.path.isabs(bench_gliner_out):
                if "benchmark_gliner" not in cfg:
                    cfg["benchmark_gliner"] = {}
                cfg["benchmark_gliner"]["output_dir"] = str(
                    (Path(config_path).parent / bench_gliner_out).resolve()
                )

            bench_g_start = time.time()
            bench_g_results = run_all_gliner_benchmarks(cfg)
            bench_g_elapsed = format_time(time.time() - bench_g_start)

            success_count = sum(1 for r in bench_g_results if r["status"] == "SUCCESS")
            results["benchmark_gliner"] = {
                "status": "SUCCESS" if success_count > 0 else "FAILED",
                "time": bench_g_elapsed,
                "output": cfg.get("benchmark_gliner", {}).get("output_dir", "N/A"),
                "models_ok": success_count,
                "models_total": len(bench_g_results),
            }

        except KeyboardInterrupt:
            print("\n[Benchmark GLiNER] Đã dừng")
            results["benchmark_gliner"] = {"status": "INTERRUPTED"}
        except Exception as e:
            print(f"\n[Benchmark GLiNER] LỖI: {e}")
            import traceback
            traceback.print_exc()
            results["benchmark_gliner"] = {"status": "FAILED", "error": str(e)}

    # ============================================================
    # BENCHMARK Classifier
    # ============================================================
    if run_benchmark:
        print_banner("BẮT ĐẦU BENCHMARK")
        try:
            from benchmark_classifier import run_all_benchmarks

            bench_start = time.time()
            bench_results = run_all_benchmarks(cfg)
            bench_elapsed = format_time(time.time() - bench_start)

            success_count = sum(1 for r in bench_results if r["status"] == "SUCCESS")
            results["benchmark"] = {
                "status": "SUCCESS" if success_count > 0 else "FAILED",
                "time": bench_elapsed,
                "output": cfg.get("benchmark", {}).get("output_dir", "N/A"),
                "models_ok": success_count,
                "models_total": len(bench_results),
            }

        except KeyboardInterrupt:
            print("\n[Benchmark] Đã dừng bởi người dùng")
            results["benchmark"] = {"status": "INTERRUPTED"}
        except Exception as e:
            print(f"\n[Benchmark] LỖI: {e}")
            import traceback
            traceback.print_exc()
            results["benchmark"] = {"status": "FAILED", "error": str(e)}

    # ============================================================
    # TÓM TẮT KẾT QUẢ CUỐI
    # ============================================================
    total_elapsed = format_time(time.time() - overall_start)
    print_banner(f"KẾT QUẢ (Tổng thời gian: {total_elapsed})")

    for task_name, result in results.items():
        status = result["status"]
        if status == "SUCCESS":
            print(f"  ✓ {task_name.upper()}")
            if "models_ok" in result:
                print(f"      Models OK : {result['models_ok']}/{result['models_total']}")
            print(f"      Output    : {result.get('output', 'N/A')}")
            print(f"      Time      : {result.get('time', 'N/A')}")
        elif status == "INTERRUPTED":
            print(f"  ⚡ {task_name.upper()} - Bị ngắt")
        else:
            print(f"  ✗ {task_name.upper()} - THẤT BẠI")
            print(f"      Lỗi: {result.get('error', 'N/A')}")
        print()

    all_success = all(r["status"] == "SUCCESS" for r in results.values())
    if all_success:
        print("Tất cả tác vụ đã hoàn thành thành công!")
    else:
        print("Một số tác vụ bị lỗi. Kiểm tra log phía trên để biết chi tiết.")

    return 0 if all_success else 1



if __name__ == "__main__":
    sys.exit(main())
