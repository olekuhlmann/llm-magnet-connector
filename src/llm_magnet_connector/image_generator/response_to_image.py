
import os
from llm_magnet_connector.llm_interface import LLMResponse
from ._annotate_imgs import annotate_images 
from ._generate_curve_images import CurveImageGenerator


class ResponseToImage:
    """
    This class is used to convert a LLM response to the images used for the next re-prompt
    
    args:
        output_dir: The directory where the images will be saved. Dir will be created if it does not exist. The images corresponding to one curve will be saved in a folder named by the image index. Each image will be saved as a PNG file.
    """
    def __init__(self, output_dir: str):
        # Ascending index for the image names (0a, 1a, ...); 1-indexed, will be incremented before use
        self.image_index = 0 
        self._output_dir = output_dir
        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        
    def response_to_image(self, response: LLMResponse) -> str:
        """
        This function converts a LLM response to the images used for the next re-prompt
        
        args:
            response: The response from the LLM model
            
        Returns:
            The path to the directory containing the generated images.
        """
        self.image_index += 1
        # create the output directory
        new_dir_path = os.path.join(self._output_dir, str(self.image_index))
        
        if os.path.exists(new_dir_path):
            print(f"[WARNING] Output directory {new_dir_path} already exists.");
        else:
            os.makedirs(new_dir_path)
        
        # generate images
        CurveImageGenerator().generate_images(new_dir_path, optimizer_params=response.optimizer_parameters, index=self.image_index)
        
        # annotate images
        annotate_images(new_dir_path,  new_dir_path)
        
        # return path
        return new_dir_path