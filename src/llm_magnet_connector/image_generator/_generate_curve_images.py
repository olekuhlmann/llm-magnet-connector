from pathlib import Path
import time
from llm_magnet_connector.llm_interface import OptimizerParameters

class CurveImageGenerator:
    """
    This class is used to take a set of optimizer parameters and generate images from the curve created by those optimizer params.
    """
    def __init__(self, logger):
        """
        Initializes the CurveImageGenerator.
        
        args:
            logger: The logger to use.
        """
        self.logger = logger
        pass
    
    def _wait_for_images(self, dir, image_names: list):
        """
        Waits for the user to manually create the images. Images must be .png.
        
        args:
            dir: The directory the user should put the images in
            image_names: List of expected image files (without .png extension).
        """
        dir_path = Path(dir)
        images_found = False
        self.logger.info(f"Waiting for images {image_names} in directory {dir}")
        while not images_found:
            # check if images are there
            images_found = all((dir_path / f"{name}.png").exists() for name in image_names)
            if not images_found:
                time.sleep(0.01)
        self.logger.info("Images found.")
            
    
    def generate_images(self, dir, optimizer_params: OptimizerParameters, index: int):
        """
        Optimizes the curve using the given optimizer parameters and creates images from the curve.
        
        args:
            dir: The directory for the output images
            optimizer_params: Optimizer parameters to use 
            index: Index for the image names.
        """
        # TODO stub implementation
        self.logger.info(f"Please apply optimizer params: {optimizer_params}")
        self._wait_for_images(dir, [f"{index}a", f"{index}b", f"{index}c"])