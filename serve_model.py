from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import OpenAIServing
from vllm.entrypoints.openai.protocol import CompletionRequest
import asyncio
import logging
import sys
import torch
import traceback
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_server.log')
    ]
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return f"RSS: {memory_info.rss / 1024 / 1024:.2f}MB, VMS: {memory_info.vms / 1024 / 1024:.2f}MB"

async def main():
    try:
        logger.info(f"Starting server process (PID: {os.getpid()})")
        logger.info(f"Initial memory usage - {get_memory_usage()}")
        
        logger.debug("Checking CUDA availability...")
        if torch.cuda.is_available():
            logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA is not available")
        
        logger.debug("Checking MPS availability...")
        if torch.backends.mps.is_available():
            logger.info("MPS (Metal Performance Shaders) is available but will not be used")
        else:
            logger.info("MPS is not available")

        # Configure engine arguments
        logger.info("Configuring engine arguments...")
        engine_args = AsyncEngineArgs(
            model="OpenGVLab/InternVL2-2B",
            trust_remote_code=True,
            dtype="float16",  # Use float16 for better memory efficiency
            max_model_len=2048,
            device="cpu",  # Force CPU usage
            worker_use_ray=False,  # Disable Ray for local deployment
            tensor_parallel_size=1  # Single device configuration
        )
        logger.debug(f"Engine arguments configured: {engine_args}")

        # Initialize the engine
        logger.info("Starting LLM engine initialization...")
        logger.info(f"Memory usage before engine init - {get_memory_usage()}")
        try:
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("LLM engine initialized successfully")
            logger.info(f"Memory usage after engine init - {get_memory_usage()}")
        except Exception as engine_error:
            logger.error("Failed to initialize LLM engine")
            logger.error(f"Error details: {str(engine_error)}")
            logger.debug("Full traceback:", exc_info=True)
            raise

        # Create OpenAI API server
        openai_serving = OpenAIServing(engine)
        
        # Start the server
        host = "0.0.0.0"
        port = 23333
        logger.info(f"Starting OpenAI-compatible API server on http://{host}:{port}")
        await openai_serving.run(host=host, port=port)

    except Exception as e:
        logger.error("Fatal error in main loop")
        logger.error(f"Error details: {str(e)}")
        logger.debug("Full traceback:")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested. Exiting...")
    except Exception as e:
        logger.error("Fatal error in main process")
        logger.error(f"Error details: {str(e)}")
        logger.debug("Full traceback:")
        logger.debug(traceback.format_exc())
        sys.exit(1)

