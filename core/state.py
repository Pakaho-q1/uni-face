import configparser
import argparse
import os
from core.config import ROOT_DIR

class StateManager:
    def __init__(self):
        self.providers = ["CPUExecutionProvider"]
        self.processors = ["swap", "restore", "color"]
        self.execution_thread_count = 4
        self.video_encoder = "h264_nvenc"
        
        self.swap_model = "inswapper_128"
        self.swap_weight = 0.65
        self.swap_boost = 128
        self.mask_types = ["box"]
        
        self.restore_model = "gfpgan_1.4"
        self.restore_weight = 1.0
        self.restore_blend = 100

        self.source_path = None
        self.target_path = None
        self.output_path = None
        self.similarity = False

    def init(self, parse_args=True):
        ini_path = ROOT_DIR / "uni-face.ini"
        config = configparser.ConfigParser()
        if os.path.exists(ini_path):
            config.read(ini_path)
            if "GLOBAL" in config:
                if "providers" in config["GLOBAL"]:
                    self._parse_providers(config["GLOBAL"]["providers"])
                if "execution_thread_count" in config["GLOBAL"]:
                    self.execution_thread_count = int(config["GLOBAL"]["execution_thread_count"])
                if "video_encoder" in config["GLOBAL"]:
                    self.video_encoder = config["GLOBAL"]["video_encoder"]
            if "PROCESSORS" in config:
                p = config["PROCESSORS"]
                if "processors" in p: self.processors = p["processors"].split()
                if "swap_model" in p: self.swap_model = p["swap_model"]
                if "swap_weight" in p: self.swap_weight = float(p["swap_weight"])
                if "swap_boost" in p: self.swap_boost = int(p["swap_boost"])
                if "mask_types" in p: self.mask_types = p["mask_types"].split()
                if "restore_model" in p: self.restore_model = p["restore_model"]
                if "restore_weight" in p: self.restore_weight = float(p["restore_weight"])
                if "restore_blend" in p: self.restore_blend = int(p["restore_blend"])

        if not parse_args:
            return

        parser = argparse.ArgumentParser(description="Uni-Face Pipeline")
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-o", "--output", required=True)
        parser.add_argument("--similarity", action="store_true")
        
        parser.add_argument("--providers", nargs="+")
        parser.add_argument("--execution_thread_count", type=int)
        parser.add_argument("--video_encoder", type=str)
        parser.add_argument("--processors", nargs="+")
        parser.add_argument("--swap_model", type=str)
        parser.add_argument("--swap_weight", type=float)
        parser.add_argument("--swap_boost", type=int)
        parser.add_argument("--mask_types", nargs="+")
        parser.add_argument("--restore_model", type=str)
        parser.add_argument("--restore_weight", type=float)
        parser.add_argument("--restore_blend", type=int)
        
        args = parser.parse_args()
        
        self.source_path = args.source
        self.target_path = args.target
        self.output_path = args.output
        self.similarity = args.similarity
        
        if args.providers: self._parse_providers(" ".join(args.providers))
        if args.execution_thread_count is not None: self.execution_thread_count = args.execution_thread_count
        if args.video_encoder: self.video_encoder = args.video_encoder
        if args.processors: self.processors = args.processors
        if args.swap_model: self.swap_model = args.swap_model
        if args.swap_weight is not None: self.swap_weight = args.swap_weight
        if args.swap_boost is not None: self.swap_boost = args.swap_boost
        if args.mask_types: self.mask_types = args.mask_types
        if args.restore_model: self.restore_model = args.restore_model
        if args.restore_weight is not None: self.restore_weight = args.restore_weight
        if args.restore_blend is not None: self.restore_blend = args.restore_blend

    def _parse_providers(self, provider_str):
        mapping = {
            "trt": "TensorrtExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "cpu": "CPUExecutionProvider"
        }
        self.providers = []
        import onnxruntime
        cache_path = os.path.join(ROOT_DIR, '.caches', onnxruntime.get_version_string())
        
        for p in provider_str.split():
            provider_name = mapping.get(p.lower())
            if not provider_name:
                continue
                
            if provider_name == 'TensorrtExecutionProvider':
                os.makedirs(cache_path, exist_ok=True)
                trt_options = {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': cache_path,
                    'trt_timing_cache_enable': True,
                    'trt_timing_cache_path': cache_path,
                    'trt_builder_optimization_level': 4
                }
                cuda_options = {
                    'cudnn_conv_algo_search': 'DEFAULT'
                }
                self.providers = [(provider_name, trt_options), ('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
                break
            elif provider_name == 'CUDAExecutionProvider':
                cuda_options = {
                    'cudnn_conv_algo_search': 'DEFAULT'
                }
                self.providers = [(provider_name, cuda_options), 'CPUExecutionProvider']
                break
            else:
                self.providers = ['CPUExecutionProvider']
                break
                
        if not self.providers:
            self.providers = ["CPUExecutionProvider"]

state = StateManager()

