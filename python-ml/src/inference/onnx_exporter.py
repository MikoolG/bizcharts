"""Export models to ONNX for optimized CPU inference.

ONNX Runtime provides 4-10x speedup for inference on CPU.
Quantization further reduces model size and improves performance.

DistilBERT achieves ~25ms per prediction on CPU with ONNX,
processing 2,500+ posts per minute.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_setfit_to_onnx(
    model_path: str | Path,
    output_path: str | Path,
    quantize: bool = True,
) -> Path:
    """Export SetFit model to ONNX format.

    The SetFit model has two components:
    1. Sentence transformer body (for embeddings)
    2. Classification head

    This exports the sentence transformer to ONNX and saves
    the classification head separately.

    Args:
        model_path: Path to trained SetFit model
        output_path: Directory for exported model
        quantize: Apply INT8 quantization

    Returns:
        Path to exported model directory
    """
    from setfit import SetFitModel

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading SetFit model from {model_path}")
    model = SetFitModel.from_pretrained(str(model_path))

    # Export sentence transformer body
    logger.info("Exporting sentence transformer to ONNX...")

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction

        # Get the model body (sentence transformer)
        body = model.model_body

        # Export to ONNX
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            body._modules["0"].auto_model.name_or_path,
            export=True,
        )

        onnx_path = output_path / "onnx"
        ort_model.save_pretrained(str(onnx_path))

        if quantize:
            _quantize_onnx_model(onnx_path, onnx_path / "quantized")

        logger.info(f"Exported ONNX model to {onnx_path}")

        # Save classification head separately
        import torch

        head_path = output_path / "classification_head.pt"
        torch.save(model.model_head.state_dict(), head_path)
        logger.info(f"Saved classification head to {head_path}")

        return output_path

    except ImportError as e:
        logger.error(f"optimum not installed: {e}")
        raise


def export_cryptobert_to_onnx(
    model_path: str | Path,
    output_path: str | Path,
    quantize: bool = True,
) -> Path:
    """Export CryptoBERT model to ONNX format.

    Args:
        model_path: Path to trained CryptoBERT model
        output_path: Directory for exported model
        quantize: Apply INT8 quantization

    Returns:
        Path to exported model directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting CryptoBERT from {model_path}")

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification

        # Export to ONNX
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            str(model_path),
            export=True,
            provider="CPUExecutionProvider",
        )

        onnx_path = output_path / "onnx"
        ort_model.save_pretrained(str(onnx_path))

        if quantize:
            _quantize_onnx_model(onnx_path, onnx_path / "quantized")

        logger.info(f"Exported ONNX model to {onnx_path}")
        return output_path

    except ImportError as e:
        logger.error(f"optimum not installed: {e}")
        raise


def _quantize_onnx_model(input_path: Path, output_path: Path):
    """Apply INT8 quantization to ONNX model.

    Args:
        input_path: Path to ONNX model
        output_path: Path for quantized model
    """
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        logger.info("Applying INT8 quantization...")

        output_path.mkdir(parents=True, exist_ok=True)

        # Load quantizer
        quantizer = ORTQuantizer.from_pretrained(str(input_path))

        # Configure quantization (dynamic for best compatibility)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

        # Quantize
        quantizer.quantize(
            save_dir=str(output_path),
            quantization_config=qconfig,
        )

        logger.info(f"Quantized model saved to {output_path}")

    except ImportError:
        logger.warning("Quantization requires optimum[onnxruntime]. Skipping.")
    except Exception as e:
        logger.warning(f"Quantization failed: {e}. Skipping.")


def benchmark_onnx_model(
    model_path: str | Path,
    num_samples: int = 100,
    batch_size: int = 1,
) -> dict:
    """Benchmark ONNX model inference speed.

    Args:
        model_path: Path to ONNX model
        num_samples: Number of samples to run
        batch_size: Batch size for inference

    Returns:
        Dictionary with benchmark results
    """
    import time

    from transformers import AutoTokenizer

    logger.info(f"Benchmarking ONNX model from {model_path}")

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification

        # Load model and tokenizer
        model = ORTModelForSequenceClassification.from_pretrained(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Generate test inputs
        test_texts = ["This is a test sentence for benchmarking the model."] * num_samples

        # Warmup
        inputs = tokenizer(test_texts[:batch_size], return_tensors="pt", padding=True)
        _ = model(**inputs)

        # Benchmark
        start_time = time.time()

        for i in range(0, num_samples, batch_size):
            batch = test_texts[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            _ = model(**inputs)

        elapsed = time.time() - start_time

        results = {
            "num_samples": num_samples,
            "batch_size": batch_size,
            "total_time_ms": elapsed * 1000,
            "avg_time_per_sample_ms": (elapsed * 1000) / num_samples,
            "samples_per_second": num_samples / elapsed,
        }

        logger.info(f"Benchmark results: {results['avg_time_per_sample_ms']:.2f}ms per sample")
        return results

    except ImportError as e:
        logger.error(f"optimum not installed: {e}")
        return {}


def main():
    """CLI for ONNX export."""
    import argparse

    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("model_path", help="Path to model")
    parser.add_argument("output_path", help="Output directory")
    parser.add_argument(
        "--type",
        choices=["setfit", "cryptobert"],
        required=True,
        help="Model type",
    )
    parser.add_argument("--no-quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after export")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.type == "setfit":
        output = export_setfit_to_onnx(
            args.model_path, args.output_path, quantize=not args.no_quantize
        )
    else:
        output = export_cryptobert_to_onnx(
            args.model_path, args.output_path, quantize=not args.no_quantize
        )

    if args.benchmark:
        benchmark_onnx_model(output / "onnx")


if __name__ == "__main__":
    main()
