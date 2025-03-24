
import os
from datetime import datetime

from llm_magnet_connector.image_generator import ResponseToImage
from llm_magnet_connector.llm_interface import AnthropicConversationManager, get_system_prompt, get_initial_prompt, OptimizerParameters
from llm_magnet_connector.orchestrator import MainOrchestrator

def create_dir_with_timestamp(base_path) -> str:
    """
    Creates a directory with a timestamp as the name in human readable format.
    
    Args:
        base_path (str): The parent directory of the new dir.
        
    Returns:
        The path to the new directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_dir = os.path.join(base_path, timestamp)
    os.makedirs(new_dir)
    return new_dir


output_dir = "runs"
output_dir = create_dir_with_timestamp(output_dir)

llm_manager = AnthropicConversationManager(max_prompts=100, context_window_limit=60000, system_prompt=get_system_prompt())
image_generator = ResponseToImage(output_dir)

max_iterations = 10
orchestrator = MainOrchestrator(llm_manager, image_generator, max_iterations)
orchestrator.run(get_initial_prompt(OptimizerParameters(9, 80, 20, -8)), "assets/temp_test_scenario2")


