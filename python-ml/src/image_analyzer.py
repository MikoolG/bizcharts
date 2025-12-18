"""Image sentiment analysis using CLIP and meme detection."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class ImageSentimentResult:
    """Result of image sentiment analysis."""

    score: float  # -1 to +1
    confidence: float  # 0 to 1
    method: str  # 'clip', 'yolo', 'fused'
    detected_elements: list[str]  # e.g., ['pink_wojak', 'text']
    ocr_text: str | None


class CLIPClassifier:
    """Zero-shot image classification using CLIP."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._labels = [
            "a meme showing financial success and gains",
            "a meme showing financial loss and despair",
            "a neutral meme about cryptocurrency",
            "a celebration meme with happy emotions",
            "a sad or crying meme showing distress",
        ]

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self.model is None:
            from transformers import CLIPModel, CLIPProcessor

            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def classify(self, image: Image.Image) -> dict:
        """
        Classify image sentiment using CLIP.

        Args:
            image: PIL Image to classify

        Returns:
            Dict with sentiment score and confidence
        """
        self._load_model()

        import torch

        inputs = self.processor(
            text=self._labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        # Map probabilities to sentiment
        # Labels 0, 3 are bullish; 1, 4 are bearish; 2 is neutral
        bullish_prob = probs[0].item() + probs[3].item()
        bearish_prob = probs[1].item() + probs[4].item()

        sentiment = bullish_prob - bearish_prob
        confidence = max(bullish_prob, bearish_prob)

        return {
            "score": sentiment,
            "confidence": confidence,
            "method": "clip",
            "probabilities": {
                "bullish": bullish_prob,
                "bearish": bearish_prob,
                "neutral": probs[2].item(),
            },
        }


class WojakDetector:
    """Detect Wojak/Pepe variants using YOLOv8."""

    # Sentiment mapping for detected variants
    VARIANT_SENTIMENT = {
        "pink_wojak": -0.8,
        "crying_wojak": -0.7,
        "green_wojak": 0.7,
        "smug_wojak": 0.3,
        "doomer": -0.6,
        "bloomer": 0.6,
        "gigachad": 0.5,
        "chad": 0.4,
        "soyjak": -0.3,
        "crying_pepe": -0.7,
        "sad_pepe": -0.5,
        "smug_pepe": 0.3,
        "happy_pepe": 0.5,
        "bobo": -0.6,
        "mumu": 0.6,
    }

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self.model is None and self.model_path:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)

    def detect(self, image_path: str | Path) -> list[dict]:
        """
        Detect Wojak/Pepe variants in image.

        Args:
            image_path: Path to image file

        Returns:
            List of detections with variant, confidence, and sentiment
        """
        if self.model is None:
            # Return empty if no model loaded
            return []

        self._load_model()

        results = self.model(str(image_path))
        detections = []

        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                if class_name in self.VARIANT_SENTIMENT:
                    detections.append(
                        {
                            "variant": class_name,
                            "confidence": float(box.conf),
                            "sentiment": self.VARIANT_SENTIMENT[class_name],
                            "bbox": box.xyxy[0].tolist(),
                        }
                    )

        return detections


class MemeOCR:
    """Extract text from meme images using PaddleOCR."""

    def __init__(self):
        self.ocr = None

    def _load_ocr(self) -> None:
        """Lazy load OCR model."""
        if self.ocr is None:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def extract_text(self, image_path: str | Path) -> str:
        """
        Extract text from image.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text as single string
        """
        self._load_ocr()

        result = self.ocr.ocr(str(image_path), cls=True)

        if not result or not result[0]:
            return ""

        texts = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.5:
                texts.append(text)

        return " ".join(texts)


class ImageAnalyzer:
    """Combined image sentiment analyzer."""

    def __init__(
        self,
        use_clip: bool = True,
        use_yolo: bool = True,
        use_ocr: bool = True,
        yolo_model_path: str | None = None,
    ):
        self.clip = CLIPClassifier() if use_clip else None
        self.yolo = WojakDetector(yolo_model_path) if use_yolo else None
        self.ocr = MemeOCR() if use_ocr else None

    def analyze(self, image_path: str | Path) -> ImageSentimentResult:
        """
        Analyze image for sentiment.

        Args:
            image_path: Path to image file

        Returns:
            ImageSentimentResult with combined analysis
        """
        image_path = Path(image_path)
        detected_elements = []
        scores = []
        ocr_text = None

        # CLIP classification
        if self.clip:
            try:
                image = Image.open(image_path).convert("RGB")
                clip_result = self.clip.classify(image)
                scores.append(
                    (clip_result["score"], clip_result["confidence"], "clip")
                )
            except Exception:
                pass

        # YOLO detection
        if self.yolo:
            try:
                detections = self.yolo.detect(image_path)
                for det in detections:
                    detected_elements.append(det["variant"])
                    scores.append(
                        (det["sentiment"], det["confidence"], "yolo")
                    )
            except Exception:
                pass

        # OCR
        if self.ocr:
            try:
                ocr_text = self.ocr.extract_text(image_path)
                if ocr_text:
                    detected_elements.append("text")
            except Exception:
                pass

        # Combine scores
        if not scores:
            return ImageSentimentResult(
                score=0.0,
                confidence=0.0,
                method="none",
                detected_elements=[],
                ocr_text=None,
            )

        # Weighted average by confidence
        total_weight = sum(conf for _, conf, _ in scores)
        if total_weight > 0:
            combined_score = sum(score * conf for score, conf, _ in scores) / total_weight
            combined_confidence = min(total_weight, 1.0)
        else:
            combined_score = 0.0
            combined_confidence = 0.0

        # Determine primary method
        methods = [m for _, _, m in scores]
        primary_method = "fused" if len(set(methods)) > 1 else methods[0]

        return ImageSentimentResult(
            score=combined_score,
            confidence=combined_confidence,
            method=primary_method,
            detected_elements=detected_elements,
            ocr_text=ocr_text,
        )


def main() -> None:
    """Test the image analyzer."""
    # This requires actual images to test
    print("ImageAnalyzer module loaded successfully")
    print("Available components:")
    print("  - CLIPClassifier: Zero-shot meme classification")
    print("  - WojakDetector: YOLOv8-based meme character detection")
    print("  - MemeOCR: PaddleOCR text extraction")
    print("\nNote: YOLOv8 model needs to be trained on Wojak/Pepe dataset")


if __name__ == "__main__":
    main()
