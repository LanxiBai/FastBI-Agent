# FastBI-Agent

FastBI-Agent is a planned AI-powered Business Intelligence Analyst Agent. The project will enable users to upload Excel or CSV datasets and request analyses in natural language. Future versions are intended to coordinate data analysis, visualization, and business-insight generation through an agent workflow.

This repository currently contains only the initial project structure and configuration. Agent workflows, data-processing features, visualizations, and the web interface have not yet been implemented.

## Technology Stack

- Python 3.11+
- LangGraph
- LangChain
- Streamlit
- Pandas and NumPy
- Plotly
- OpenPyXL
- OpenAI-compatible APIs, including OpenAI and DeepSeek

## Development Roadmap

- Define application configuration and model-provider abstraction.
- Implement safe CSV and Excel ingestion and validation.
- Build reusable data-analysis and visualization tools.
- Design the LangGraph agent workflow and state model.
- Add a Streamlit interface for file upload and natural-language requests.
- Generate transparent business insights with supporting charts and summaries.
- Add automated tests, examples, and deployment documentation.
