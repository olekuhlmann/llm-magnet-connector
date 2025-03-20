from llm_magnet_connector.image_generator import ResponseToImage
from llm_magnet_connector.llm_interface import LLMResponse, OptimizerParameters, BadnessCriteria

output_dir= "runs/001"

response = LLMResponse(badnessCriteria=BadnessCriteria(True, True, True, True), optimizer_parameters=OptimizerParameters(7, 80, 25, -6))

res2img = ResponseToImage(output_dir)
res2img.response_to_image(response)
res2img.response_to_image(response)