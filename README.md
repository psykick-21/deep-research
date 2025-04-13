# Deep Research

A sophisticated research automation tool that leverages LangChain and LangGraph to conduct in-depth research on any topic. The system uses multiple AI agents working together to gather information, analyze data, and generate comprehensive research reports.

## Features

- Automated research workflow using multiple specialized AI agents
- Deep web search capabilities using Tavily
- Structured report generation with customizable sections
- Human-in-the-loop feedback system
- Configurable research parameters (depth, temperature, etc.)
- Detailed logging and progress tracking

## Prerequisites

- Python 3.12 or higher
- Poetry for dependency management
- API keys for:
  - OpenAI
  - Anthropic
  - Google AI
  - Tavily

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-research.git
cd deep-research
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
TAVILY_API_KEY=your_tavily_key
```

## Usage

1. Activate the Poetry virtual environment:
```bash
poetry shell
```

2. Run the research workflow:
```bash
python main.py
```

The default configuration will research "LLM Benchmarking" with a focus on comparing different LLMs' capabilities, performance, and effectiveness. You can modify the topic and outline in `main.py` to research any subject of your choice.

## Project Structure

- `deep_research/`: Main package directory
  - `nodes.py`: Contains the implementation of various research agents
  - `graph.py`: Defines the LangGraph workflow
  - `prompts.py`: Contains the prompts used by different agents
  - `struct.py`: Defines the data structures used in the workflow
  - `state.py`: Manages the state of the research process
  - `utils.py`: Utility functions
  - `configuration.py`: Configuration settings

- `logs/`: Contains detailed logs of the research process
- `reports/`: Stores the generated research reports
- `main.py`: Entry point of the application

## Configuration

You can customize the research process by modifying the following parameters in `main.py`:

- `max_queries`: Maximum number of search queries to perform
- `search_depth`: Depth of the research
- `num_reflections`: Number of reflection cycles
- `temperature`: Controls the creativity of the AI responses

## License

This project is licensed under the terms included in the LICENSE file.

## Author

Gaurav Sharma (psykickai@gmail.com)
