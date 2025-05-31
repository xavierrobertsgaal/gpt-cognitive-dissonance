# Replication and extension of Lehr et al. (2025)

# Imports and setup
import pandas as pd
import openai
from openai import OpenAI
from typing import Dict, List, Tuple, Optional, Protocol
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass, field, replace
import logging
from pathlib import Path
import json
import os
from dotenv import load_dotenv
import argparse
import backoff
import concurrent.futures
import random

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API setup
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)  # note: depends on OPENAI_API_KEY environment variable set in .env file


@dataclass(frozen=True)
class ConditionConfig:
    """Configuration for a specific experimental condition."""

    name: str
    first_prompt: str
    interim_prompt: str
    evaluative_questions: Dict[str, str]

    def __post_init__(self):
        """Validate the condition configuration."""
        if not self.name:
            raise ValueError("Condition name cannot be empty")
        if not self.first_prompt:
            raise ValueError("First prompt cannot be empty")
        if not self.interim_prompt:
            raise ValueError("Interim prompt cannot be empty")
        if not self.evaluative_questions:
            raise ValueError("Evaluative questions cannot be empty")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the entire experiment."""

    model: str
    temperature: float
    output_dir: Path
    experiment_version: str
    evaluative_response_format: str
    conditions: Dict[str, ConditionConfig]
    samples_per_condition: int
    instructions: Optional[str] = None

    @classmethod
    def from_json(cls, json_path: str) -> "ExperimentConfig":
        """Create an experiment configuration from a JSON file.

        Args:
            json_path: Path to the JSON configuration file.

        Returns:
            An ExperimentConfig instance populated from the JSON file.

        Raises:
            FileNotFoundError: If the JSON file doesn't exist.
            json.JSONDecodeError: If the JSON file is invalid.
            ValueError: If required fields are missing or invalid.
        """
        with open(json_path, "r") as f:
            config_dict = json.load(f)

        # Create condition configs
        condition_configs = {}
        for name, condition_data in config_dict["conditions"].items():
            condition_configs[name] = ConditionConfig(
                name=name,
                first_prompt=condition_data["first_prompt"],
                interim_prompt=condition_data["interim_prompt"],
                evaluative_questions=condition_data["evaluative_questions"],
            )

        return cls(
            model=config_dict["model"],
            temperature=config_dict["temperature"],
            output_dir=Path(config_dict["output_dir"]),
            experiment_version=config_dict["experiment_version"],
            evaluative_response_format=config_dict["evaluative_response_format"],
            conditions=condition_configs,
            samples_per_condition=config_dict["samples_per_condition"],
            instructions=(
                config_dict["instructions"] if "instructions" in config_dict else None
            ),
        )


# Helper functions

# Key input parameters to OpenAI's Responses API:
#   input: Text input to the model.
#   model: "gpt-3.5-turbo" or "gpt-4o"
#   temperature: 0.0-2.0; default is 1.0
#   previous_response_id: Manages conversation state by linking the response to the previous message
#   metadata: Arbitrary metadata to include with the request. 16 key-value pairs max. Keys are up to 64 characters, values are up to 512 characters.
# Output structure:
#   created_at: Unix timestamp of when the response was created (in seconds)
#   error: Error message if the request failed, otherwise null
#   id: Unique identifier for the response. Key for previous_response_id
#   output_text: SDK convenience property that returns all the text output by the model.


def safe_api_call(func):
    """Decorator to safely call OpenAI API functions.

    Args:
        func: The function to decorate.

    Returns:
        The response from the API call, or None if the call fails.
    """

    def wrapper(*args, **kwargs):
        try:
            response = func(*args, **kwargs)
            if not response:
                raise Exception("No response returned")
            return response
        except Exception as e:
            logger.error(f"Error in API call: {str(e)}")
            if hasattr(e, "response"):
                logger.error(f"API Response: {e.response}")
            return None

    return wrapper


@backoff.on_exception(
    backoff.expo,
    openai.RateLimitError,
    max_tries=5,
    max_time=300,
    jitter=backoff.full_jitter,
)
@safe_api_call
def create_response(
    model: str,
    input: str,
    temperature: float = 1.0,
    metadata: Dict = None,
    previous_response_id: str = None,
    instructions: str = None,
):
    """Create a response from the OpenAI API with exponential backoff for rate limits."""
    return client.responses.create(
        model=model,
        input=input,
        temperature=temperature,
        metadata=metadata,
        previous_response_id=previous_response_id,
        instructions=instructions,
    )


@dataclass
class TrialConfig:
    """Configuration for a single trial."""

    condition: ConditionConfig
    experiment_config: ExperimentConfig
    trial_id: str
    timestamp: str

    @property
    def metadata(self) -> Dict[str, str]:
        """Compute metadata for the trial.

        Returns:
            A dictionary containing trial metadata including condition name,
            trial ID, timestamp, and experiment version.
        """
        return {
            "condition": self.condition.name,
            "trial_id": self.trial_id,
            "timestamp": self.timestamp,
            "experiment_version": self.experiment_config.experiment_version,
        }


def create_condition_configs(
    experiment_config: ExperimentConfig,
) -> List[ConditionConfig]:
    """Create condition configurations from the experiment config."""
    return list(experiment_config.conditions.values())


class ResponseProtocol(Protocol):
    """Protocol for API responses."""

    id: str
    output_text: str


@dataclass
class TrialResult:
    """Result from a single trial."""

    trial_id: str
    condition: str
    timestamp: str
    essay: str
    interim: str
    evaluative_responses: Dict[str, str]
    error: Optional[str] = None
    scores: Optional[Dict[str, Tuple[float, str]]] = None

    def to_dict(self) -> Dict:
        """Convert trial result to dictionary format."""
        record = {
            "trial_id": self.trial_id,
            "condition": self.condition,
            "timestamp": self.timestamp,
            "essay": self.essay,
            "interim": self.interim,
            "error": self.error,
        }

        if self.evaluative_responses:
            for question, response in self.evaluative_responses.items():
                score, explanation = self._score_response(response)
                record[f"{question}_score"] = score
                record[f"{question}_explanation"] = explanation
                record[f"{question}_raw_response"] = response

        return record

    @staticmethod
    def _score_response(response: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract score and explanation from a response.

        Handles:
        - Markdown bold before "Answer"
        - Extracts first float from text (e.g., "3.5 - Slightly bad")
        - Standard "Answer: X" format
        - Hyphenated ranges (e.g., "3-4" or "3–4") by averaging
        """
        try:
            lines = response.split("\n")
            # Remove markdown bold
            lines = [line.replace("**", "") for line in lines]
            # Look for "Answer: X" format
            answer_line = next(
                (line for line in lines if line.startswith("Answer: ")),
                None,
            )

            if answer_line:
                score_text = answer_line.split(": ")[1].split()[
                    0
                ]  # Get first token after colon
                # Check for hyphenated range
                if "-" in score_text or "–" in score_text:
                    try:
                        # Replace en dash with regular hyphen for consistent handling
                        score_text = score_text.replace("–", "-")
                        num1, num2 = map(float, score_text.split("-"))
                        score = (num1 + num2) / 2
                    except ValueError:
                        score = None
                else:
                    try:
                        score = float(score_text)
                    except ValueError:
                        score = None
            else:
                score = None

            # Look for "Explanation: X" format
            explanation = next(
                (
                    line.split(": ")[1]
                    for line in lines
                    if line.startswith("Explanation: ")
                ),
                None,
            )

            if score is None or explanation is None:
                logger.warning(
                    f"Could not parse score or explanation from response: {response}"
                )

            return score, explanation
        except Exception as e:
            logger.error(f"Error scoring response: {e}")
            return None, None


class ExperimentRunner:
    """Handles running the experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = min(
            32, (os.cpu_count() or 1) * 4
        )  # Reasonable default for API calls
        self._executor = None

    def __enter__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)

    def create_trial_config(self, condition: ConditionConfig, idx: int) -> TrialConfig:
        """Create a trial configuration."""
        return TrialConfig(
            condition=condition,
            experiment_config=self.config,
            trial_id=f"{condition.name}_{idx}",
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S%f"),
        )

    def _create_response(
        self,
        model: str,
        input: str,
        temperature: float,
        metadata: Dict,
        previous_response_id: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> Optional[ResponseProtocol]:
        """Create a response from the OpenAI API."""
        return create_response(
            model=model,
            input=input,
            temperature=temperature,
            metadata=metadata,
            previous_response_id=previous_response_id,
            instructions=instructions,
        )

    def _get_evaluative_responses(
        self, config: TrialConfig, previous_response_id: str
    ) -> Dict[str, str]:
        """Get evaluative responses for a trial, maintaining conversation state."""
        responses = {}

        # For each evaluative question, combine it with the interim prompt and response format
        for question_key, question in config.condition.evaluative_questions.items():
            # Combine prompts while maintaining conversation state
            full_prompt = f"{question}\n\n{self.config.evaluative_response_format}"

            # Use the essay response ID to maintain conversation state
            response = self._create_response(
                model=self.config.model,
                input=full_prompt,
                temperature=self.config.temperature,
                metadata={
                    **config.metadata,
                    "question_key": question_key,
                    "stage": "evaluative_response",
                },
                previous_response_id=previous_response_id,
                instructions=self.config.instructions,
            )

            if response:
                responses[question_key] = response.output_text

        return responses

    def run_trial(self, config: TrialConfig) -> Optional[TrialResult]:
        """Run a single trial."""
        try:
            # First get the essay response
            essay_response = self._create_response(
                model=self.config.model,
                input=config.condition.first_prompt,
                temperature=self.config.temperature,
                metadata={**config.metadata, "stage": "essay"},
                instructions=self.config.instructions,
            )

            if not essay_response:
                logger.error(
                    f"Failed to get essay response for trial {config.trial_id}"
                )
                return None

            # Then get interim response, maintaining conversation state
            interim_response = self._create_response(
                model=self.config.model,
                input=config.condition.interim_prompt,
                temperature=self.config.temperature,
                metadata={**config.metadata, "stage": "interim"},
                previous_response_id=essay_response.id,
            )

            if not interim_response:
                logger.error(
                    f"Failed to get interim response for trial {config.trial_id}"
                )
                return None

            # Then get evaluative responses, maintaining conversation state
            evaluative_responses = self._get_evaluative_responses(
                config, interim_response.id
            )

            # Create result even if some questions failed
            return TrialResult(
                trial_id=config.trial_id,
                condition=config.condition.name,
                timestamp=config.timestamp,
                essay=essay_response.output_text,
                interim=interim_response.output_text,
                evaluative_responses=evaluative_responses,
            )

        except Exception as e:
            logger.error(f"Error in trial {config.trial_id}: {e}")
            return None

    def run_experiment(
        self, conditions: List[ConditionConfig]
    ) -> Optional[pd.DataFrame]:
        """Run the full experiment with parallel processing."""
        all_results = []
        experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Shuffle conditions for random order
        random.shuffle(conditions)

        total_samples = len(conditions) * self.config.samples_per_condition
        with tqdm(total=total_samples, desc="Overall Progress", position=0) as pbar:
            for condition in conditions:
                pbar.set_description(f"Running {condition.name}")
                condition_results = []

                # Create all trial configs for this condition
                trial_configs = [
                    self.create_trial_config(condition, sample_idx)
                    for sample_idx in range(self.config.samples_per_condition)
                ]

                # Run trials in parallel
                future_to_config = {
                    self._executor.submit(self.run_trial, config): config
                    for config in trial_configs
                }

                for future in concurrent.futures.as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        if result:
                            condition_results.append(result)
                            logger.info(f"Completed trial {config.trial_id}")
                        else:
                            logger.error(f"Failed trial {config.trial_id}")
                    except Exception as e:
                        logger.error(f"Error in trial {config.trial_id}: {e}")

                    pbar.update(1)

                all_results.extend(condition_results)

                # Save partial results after each condition
                if condition_results:
                    self._save_partial_results(
                        condition_results, experiment_time, condition.name
                    )

        if all_results:
            return self._save_final_results(all_results, experiment_time)
        return None

    def _save_partial_results(
        self, results: List[TrialResult], experiment_time: str, condition: str
    ) -> None:
        """Save partial results for a condition."""
        df = pd.DataFrame([r.to_dict() for r in results])
        df["model"] = self.config.model
        df["temperature"] = self.config.temperature
        df.to_csv(
            self.config.output_dir
            / f"{experiment_time}_intermediate_results_{condition}.csv",
            index=False,
        )
        logger.info(f"Saved partial results for condition {condition}")

    def _save_final_results(
        self, results: List[TrialResult], experiment_time: str
    ) -> pd.DataFrame:
        """Save and return final results."""
        df = pd.DataFrame([r.to_dict() for r in results])
        df["model"] = self.config.model
        df["temperature"] = self.config.temperature
        df.to_csv(
            self.config.output_dir / f"{experiment_time}_full_experiment_results.csv",
            index=False,
        )
        logger.info(f"Saved final results with {len(results)} trials")
        return df

    def run_experiment_interleaved(
        self, conditions: List[ConditionConfig]
    ) -> Optional[pd.DataFrame]:
        """Run the experiment with trials interleaved across conditions."""
        all_results = []
        experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create all trial configs for all conditions
        all_trial_configs = []
        for condition in conditions:
            for sample_idx in range(self.config.samples_per_condition):
                all_trial_configs.append(
                    self.create_trial_config(condition, sample_idx)
                )

        # Shuffle all trial configs
        random.shuffle(all_trial_configs)

        total_samples = len(all_trial_configs)
        with tqdm(total=total_samples, desc="Overall Progress", position=0) as pbar:
            # Run trials in parallel
            future_to_config = {
                self._executor.submit(self.run_trial, config): config
                for config in all_trial_configs
            }

            for future in concurrent.futures.as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                        logger.info(f"Completed trial {config.trial_id}")
                    else:
                        logger.error(f"Failed trial {config.trial_id}")
                except Exception as e:
                    logger.error(f"Error in trial {config.trial_id}: {e}")

                pbar.update(1)

        if all_results:
            return self._save_final_results(all_results, experiment_time)
        return None


def main():
    """Run the experiment from command line."""
    parser = argparse.ArgumentParser(
        description="Run the cognitive dissonance experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration JSON file (default: config.json)",
    )
    parser.add_argument(
        "--samples-per-condition",
        type=int,
        help="Number of samples to run per condition (overrides config file)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel workers (default: min(32, CPU_COUNT * 4))",
    )
    parser.add_argument(
        "--interleaved",
        action="store_true",
        help="Run trials in random order across all conditions",
    )

    args = parser.parse_args()

    # Load experiment configuration
    config = ExperimentConfig.from_json(args.config)

    # Override samples per condition if specified
    if args.samples_per_condition is not None:
        config = replace(config, samples_per_condition=args.samples_per_condition)

    # Create condition configurations
    conditions = create_condition_configs(config)

    # Create and run experiment with proper resource management
    with ExperimentRunner(config) as runner:
        if args.max_workers is not None:
            runner.max_workers = args.max_workers
        if args.interleaved:
            results_df = runner.run_experiment_interleaved(conditions)
        else:
            results_df = runner.run_experiment(conditions)


if __name__ == "__main__":
    main()


# Mock up calling create_response with gpt-4o and chatgpt-4o-latest

create_response(
    model="gpt-4o",
    input="{Experiment Text}",
    temperature=1.0,
    metadata={"test": "test"},
    previous_response_id="test",
)

create_response(
    model="chatgpt-4o-latest",
    input="{Experiment Text}",
    temperature=1.0,
    metadata={"test": "test"},
    previous_response_id="test",
)
