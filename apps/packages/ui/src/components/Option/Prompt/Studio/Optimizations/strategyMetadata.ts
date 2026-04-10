import type { StrategyInfo } from "@/services/prompt-studio"

export const defaultOptimizationStrategies: StrategyInfo[] = [
  {
    name: "iterative",
    display_name: "Iterative",
    description: "Iteratively refine the prompt based on feedback",
    supported_params: ["max_iterations", "learning_rate"],
    default_params: { max_iterations: 10 },
    requires_test_cases: true,
    supports_early_stopping: true
  },
  {
    name: "mipro",
    display_name: "MIPRO (Advanced Optimizer)",
    description:
      "Systematically rewrites instructions across multiple rounds to find the best-performing version. Good for complex prompts.",
    supported_params: ["max_iterations"],
    default_params: { max_iterations: 5 },
    requires_test_cases: true,
    supports_early_stopping: true
  },
  {
    name: "bootstrap",
    display_name: "Bootstrap (Auto-generate examples)",
    description:
      "Automatically generates example input/output pairs from your test cases to improve the prompt. Fast and simple.",
    supported_params: ["max_iterations"],
    default_params: { max_iterations: 3 },
    requires_test_cases: true,
    supports_early_stopping: false
  },
  {
    name: "genetic",
    display_name: "Genetic Algorithm",
    description:
      "Combines parts of different prompt variations and evolves them over generations. Explores widely but takes longer.",
    supported_params: ["population_size", "max_iterations"],
    default_params: { population_size: 10, max_iterations: 20 },
    requires_test_cases: true,
    supports_early_stopping: true
  },
  {
    name: "beam_search",
    display_name: "Beam Search",
    description:
      "Tests multiple prompt variations side-by-side and keeps the best performers. Balanced speed and quality.",
    supported_params: ["beam_width", "max_iterations"],
    default_params: { beam_width: 3, max_iterations: 10 },
    requires_test_cases: true,
    supports_early_stopping: true
  },
  {
    name: "random_search",
    display_name: "Random Search",
    description:
      "Tries many random prompt rewrites and picks the best one. Simple but can miss subtle improvements.",
    supported_params: ["max_iterations"],
    default_params: { max_iterations: 20 },
    requires_test_cases: true,
    supports_early_stopping: false
  }
]
