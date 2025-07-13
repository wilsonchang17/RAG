import os
from octotools.tools.base import BaseTool
from octotools.engine.factory import create_llm_engine

class Analysis_Result_Validator_Tool(BaseTool):
    require_llm_engine = True  # This tool will use an LLM for validation.

    def __init__(self, model_string="gpt-4o"):
        super().__init__(
            tool_name="Analysis_Result_Validator_Tool",
            tool_description=(
                "A tool that validates anomaly analysis outputs or SOPs. "
                "It checks if the analysis results are correct given the data and if the proposed solution addresses the issues. "
                "Use this after anomaly detection to verify correctness."
            ),
            tool_version="1.0.0",
            input_types={
                "content": (
                    "str - The text content to validate. This could be an anomaly analysis report, "
                    "including identified anomalies and any recommended SOP or answer."
                ),
                "original_query": (
                    "str - (Optional) The original user query or context for the analysis, for reference."
                )
            },
            output_type="str - A verification report or conclusion about the correctness of the analysis.",
            demo_commands=[
                {
                    "command": 'result = tool.execute(content="Anomaly Analysis: Temperature spiked above threshold at 14:30, which indicates a cooling failure. SOP: Check coolant levels and reboot the cooling system.", original_query="Was there an abnormal temperature event?")',
                    "description": "Validate an anomaly analysis result and its SOP against the original query."
                },
                {
                    "command": 'result = tool.execute(content="SOP: To resolve the pressure anomaly, recalibrate the sensor and inspect the valve seal.")',
                    "description": "Validate an SOP for an anomaly (without explicit query provided)."
                }
            ],
            user_metadata={
                "limitation": (
                    "This validator uses an LLM to check logical consistency and correctness. It may not catch subtle domain-specific errors or numerical inaccuracies without sufficient context. "
                    "Always double-check critical decisions."
                ),
                "best_practice": (
                    "Use the Analysis_Result_Validator_Tool after generating an analysis/SOP to double-check its validity.\n"
                    "1) Provide the anomaly analysis output or SOP text as 'content'. Include the original question in 'original_query' if available for better context.\n"
                    "2) The tool will return a message stating if the analysis seems correct and addresses the question, or point out possible issues.\n"
                    "3) Treat the response as a guidance – if it flags issues, you may need to refine the analysis or gather more data."
                )
            }
        )
        self.model_string = model_string

    def execute(self, content, original_query=None):
        print(f"Initializing Analysis Result Validator with model: {self.model_string}")
        llm_engine = create_llm_engine(model_string=self.model_string, is_multimodal=False)
        try:
            # Construct a prompt to guide the LLM for verification
            if original_query:
                prompt = (
                    "You are a validation assistant. Below is an anomaly analysis result and the original question.\n\n"
                    f"Original Question:\n{original_query}\n\n"
                    f"Analysis Result:\n{content}\n\n"
                    "Please verify whether the analysis correctly identifies the anomalies in the data and whether the recommended solution (if any) addresses the question. "
                    "Point out any errors or unsupported claims, or confirm that everything looks correct and justified."
                )
            else:
                prompt = (
                    "You are a validation assistant. Below is an anomaly analysis result (and/or SOP) to verify.\n\n"
                    f"Analysis Content:\n{content}\n\n"
                    "Please check if the analysis results are logically correct (e.g., anomalies truly indicated, conclusions follow from data) and if any proposed SOP/solution is appropriate. "
                    "Highlight any inconsistencies or confirm if it looks correct."
                )
            # Run the LLM on the prompt
            validation_response = llm_engine(prompt)
            return validation_response
        except Exception as e:
            return f"Error during validation: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

# If run as a script, demonstrate usage
if __name__ == "__main__":
    tool = Analysis_Result_Validator_Tool()
    example_content = "Anomaly Analysis: The pressure sensor reading exceeded the 6-sigma threshold at timestamp 2025-07-01 14:30, indicating a potential leak. SOP: Inspect the pipeline for leaks and recalibrate the sensor. The analysis addresses the question of pressure anomalies."
    example_query = "Were there any pressure anomalies on 2025-07-01 and how to fix them?"
    result = tool.execute(content=example_content, original_query=example_query)
    print("Validation Result:")
    print(result)