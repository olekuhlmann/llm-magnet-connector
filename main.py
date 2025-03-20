from llm_magnet_connector.image_generator import ResponseToImage
from llm_magnet_connector.llm_interface import LLMResponse, OptimizerParameters, BadnessCriteria
from llm_magnet_connector.llm_interface import get_initial_prompt, get_reprompt

output_dir= "runs/001"

response = LLMResponse(badnessCriteria=BadnessCriteria(True, True, True, True), optimizer_parameters=OptimizerParameters(7, 80, 25, -6))

res2img = ResponseToImage(output_dir)
res2img.response_to_image(response)
res2img.response_to_image(response)

print("-------------------")
print(get_initial_prompt(OptimizerParameters(7, 80, 25, -6)))
print("-------------------")
print(get_reprompt(OptimizerParameters(7, 80, 25.5, -6), 1))
print("-------------------")