import numpy
import boto3
import json
import os
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, TypedDict, List, Iterator
from botocore.exceptions import ClientError
from botocore.config import Config
import io
import re
from dateutil import parser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from contextlib import contextmanager

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(request_id)s] %(message)s'
)
logger = logging.getLogger(__name__)

class LoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f'[RequestID: {self.extra["request_id"]}] {msg}', kwargs

# Configuration Management
class AppConfig:
    """Centralized configuration management"""
    def __init__(self):
        self.face_match_threshold = float(os.environ.get("FACE_MATCH_THRESHOLD", "90"))
        self.quality_threshold = float(os.environ.get("QUALITY_THRESHOLD", "80"))
        self.s3_bucket_name = os.environ.get("S3_BUCKET_NAME", "*****")
        self.result_bucket_name = os.environ.get("RESULT_BUCKET_NAME", "*****")
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_retries = int(os.environ.get("MAX_RETRIES", "3"))
        self.chunk_size = int(os.environ.get("CHUNK_SIZE", "1048576")) 
        self.max_workers = int(os.environ.get("MAX_WORKERS", "3"))
        self.cache_ttl = int(os.environ.get("CACHE_TTL", "3600"))  

config = AppConfig()

# Metrics Collection
@dataclass
class ProcessingMetrics:
    request_id: str
    start_time: float
    end_time: float
    document_type: str
    processing_steps: List[Dict[str, float]]
    success: bool
    error: Optional[str] = None

class MetricsCollector:
    def __init__(self):
        self.metrics: List[ProcessingMetrics] = []

    def add_metric(self, metric: ProcessingMetrics):
        self.metrics.append(metric)
        self._log_metric(metric)

    def _log_metric(self, metric: ProcessingMetrics):
        total_time = metric.end_time - metric.start_time
        logger.info(
            f"Document processing metrics - "
            f"RequestID: {metric.request_id}, "
            f"Type: {metric.document_type}, "
            f"Total Time: {total_time:.2f}s, "
            f"Success: {metric.success}, "
            f"Steps: {metric.processing_steps}"
        )

metrics_collector = MetricsCollector()

# Enhanced error handling with custom exceptions
class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}

class ProcessingError(Exception):
    """Custom exception for processing errors"""
    def __init__(self, message: str, retry_allowed: bool = True):
        super().__init__(message)
        self.retry_allowed = retry_allowed

# Resource management
@contextmanager
def managed_resource():
    """Context manager for resource cleanup"""
    try:
        yield
    finally:
        cv2.destroyAllWindows()

# Enhanced AWS clients with retry configuration - Just in case "Aws can f**k up 😂"
def create_aws_client(service_name: str) -> Any:
    """Create AWS client with retry configuration"""
    config = Config(
        retries=dict(
            max_attempts=AppConfig().max_retries
        ),
        connect_timeout=5,
        read_timeout=30
    )
    return boto3.client(service_name, config=config)

s3 = create_aws_client("s3")
rekognition = create_aws_client("rekognition")
textract = create_aws_client("textract")

def cache_result(ttl_seconds: int = 3600):
    def decorator(func):
        cache = {}
        
        def wrapper(*args, **kwargs):
            key = hashlib.md5(str((args, kwargs)).encode()).hexdigest()
            now = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        
        return wrapper
    return decorator

class DocumentProcessor:
    """Enhanced document processing with multiple extraction methods"""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.logger = LoggerAdapter(logger, {"request_id": request_id})
        self.metrics = []

    @contextmanager
    def timing(self, step_name: str):
        """Context manager for timing processing steps"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            self.metrics.append({
                "step": step_name,
                "duration": end_time - start_time
            })

    @cache_result(ttl_seconds=config.cache_ttl)
    def preprocess_image(self, image_bytes: bytes) -> numpy.ndarray:
        """Enhanced image preprocessing with caching"""
        with self.timing("image_preprocessing"):
            try:
                nparr = numpy.frombuffer(image_bytes, numpy.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                max_dimension = 2000
                height, width = image.shape[:2]
                if height > max_dimension or width > max_dimension:
                    scale = max_dimension / max(height, width)
                    image = cv2.resize(image, None, fx=scale, fy=scale)

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
                denoised = cv2.fastNlMeansDenoising(binary)
                
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                angle = self._get_skew_angle(enhanced)
                if abs(angle) > 0.5:
                    enhanced = self._rotate_image(enhanced, angle)
                
                return enhanced
                
            except Exception as e:
                self.logger.error(f"Image preprocessing failed: {str(e)}")
                raise ProcessingError(f"Image preprocessing failed: {str(e)}")
            finally:
                # Cleanup
                if 'nparr' in locals():
                    del nparr

    def _get_skew_angle(self, image: numpy.ndarray) -> float:
        """Calculate skew angle of the image"""
        coords = numpy.column_stack(numpy.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        return -angle

    def _rotate_image(self, image: numpy.ndarray, angle: float) -> numpy.ndarray:
        """Rotate image by given angle"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def extract_text_with_fallback(self, image_bytes: bytes, key: str) -> str:
        """Enhanced text extraction with multiple fallback methods"""
        with self.timing("text_extraction"):
            text = ""
            errors = []
            
            # Try AWS Textract first
            try:
                response = textract.detect_document_text(
                    Document={
                        'S3Object': {
                            'Bucket': config.s3_bucket_name,
                            'Name': key
                        }
                    }
                )
                text = ' '.join([
                    block['Text'] 
                    for block in response['Blocks'] 
                    if block['BlockType'] == 'LINE'
                ])
            except Exception as e:
                errors.append(f"Textract failed: {str(e)}")
            
            # If Textract fails or returns no text, try OCR
            if not text.strip():
                try:
                    enhanced_image = self.preprocess_image(image_bytes)
                    text = pytesseract.image_to_string(enhanced_image)
                except Exception as e:
                    errors.append(f"OCR failed: {str(e)}")
            
            # If both methods fail, raise error
            if not text.strip():
                error_msg = "; ".join(errors)
                raise ProcessingError(
                    f"Text extraction failed with all methods: {error_msg}"
                )
            
            return text

class KYCProcessor:
    """Enhanced KYC document processing operations"""
    
    def __init__(self, bucket: str, user_id: str, request_id: str):
        self.bucket = bucket
        self.user_id = user_id
        self.request_id = request_id
        self.logger = LoggerAdapter(logger, {"request_id": request_id})
        self.doc_processor = DocumentProcessor(request_id)
        self.start_time = time.time()
        self.metrics = []

    def process_documents_concurrent(
        self, 
        documents: List[Dict[str, str]]
    ) -> List[ProcessingResult]:
        """Process multiple documents concurrently"""
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []
            for doc in documents:
                future = executor.submit(
                    self.process_document,
                    doc['key'],
                    doc['docType']
                )
                futures.append(future)
            
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Document processing failed: {str(e)}")
                    results.append({"error": str(e)})
            
            return results

    def process_document(self, key: str, docType: str) -> ProcessingResult:
        """Enhanced document processing with metrics and validation"""
        start_time = time.time()
        processing_steps = []
        
        try:
            self._validate_input(key, docType)
            
            # Process document
            with self.timing("document_processing"):
                result = self._process_document_internal(key, docType)
            
            end_time = time.time()
            metrics = ProcessingMetrics(
                request_id=self.request_id,
                start_time=start_time,
                end_time=end_time,
                document_type=docType,
                processing_steps=processing_steps,
                success=True
            )
            metrics_collector.add_metric(metrics)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            
            # Record failure metrics during execution
            end_time = time.time()
            metrics = ProcessingMetrics(
                request_id=self.request_id,
                start_time=start_time,
                end_time=end_time,
                document_type=docType,
                processing_steps=processing_steps,
                success=False,
                error=str(e)
            )
            metrics_collector.add_metric(metrics)
            
            raise

    def _validate_input(self, key: str, docType: str) -> None:
        """Validate input parameters"""
        if not key:
            raise ValidationError("Document key is required")
        
        if not docType:
            raise ValidationError("Document type is required")
        
        if docType not in ["front", "back", "selfie"]:
            raise ValidationError(f"Invalid document type: {docType}")

    @contextmanager
    def timing(self, step_name: str):
        """Context manager for timing processing steps"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            self.metrics.append({
                "step": step_name,
                "duration": end_time - start_time
            })

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Enhanced AWS Lambda handler with comprehensive error handling"""
    request_id = str(uuid.uuid4())
    logger_adapter = LoggerAdapter(logger, {"request_id": request_id})
    
    try:
        logger_adapter.info("Starting document processing")
        
        # Validate input
        validate_input(event)
        
        if "Records" in event:
            # S3 event trigger
            record = event["Records"][0]
            bucket = record["s3"]["bucket"]["name"]
            key = record["s3"]["object"]["key"]
            user_id, docType = parse_key(key)
        else:
            # Direct invocation - key setup
            bucket = config.s3_bucket_name
            key = event["fileKey"]
            user_id = event["userId"]
            docType = event["docType"].lower()
        
        # Processing document
        processor = KYCProcessor(bucket, user_id, request_id)
        result = processor.process_document(key, docType)
        
        # Storieng result
        store_result(user_id, docType, result)
        
        logger_adapter.info("Document processing completed successfully")
        return create_response(200, result)
        
    except ValidationError as e:
        logger_adapter.warning(f"Validation error: {str(e)}")
        return create_response(400, {
            "error": str(e),
            "details": getattr(e, 'details', None)
        })
    except ProcessingError as e:
        logger_adapter.error(f"Processing error: {str(e)}")
        return create_response(422, {
            "error": str(e),
            "retryAllowed": getattr(e, 'retry_allowed', True)
        })
    except Exception as e:
        logger_adapter.error("Unexpected error", exc_info=True)
        return create_response(500, {"error": "Internal server error"})

def validate_input(event: Dict[str, Any]) -> None:
    required_fields = {
        "direct": ["userId", "docType", "fileKey"],
        "s3": ["Records"]
    }
    
    if "Records" in event:
        event_type = "s3"
    else:
        event_type = "direct"
    
    missing_fields = [
        field for field in required_fields[event_type]
        if field not in event
    ]
    
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}"
        )

if __name__ == "__main__":
    try:
        test_event = {
            "userId": "test-user-123",
            "docType": "front",
            "fileKey": "test-user-123/front.jpg"
        }
        test_context = {}
        
        result = lambda_handler(test_event, test_context)
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error("Test execution failed", exc_info=True)