# LLM Magnet Connector

This application automates connecting two parts of a magnet model using LLMs.

This project is part of a collaboration between CERN and Cleverdist SA under the FCC-ee HTS4 research project at CERN.

## Usage
To use this application, follow these steps.

1. Clone this repository:
```sh
git clone https://github.com/ThatTilux/llm-magnet-connector.git
cd llm-magnet-connector
```  

2. Create a virtual environment:
```sh
python -m venv venv
source venv/bin/activate # On Linux
# or
venv\Scripts\activate.bat # On Windows
```

3. Install the package in editable mode:
```sh
pip install -e .
```

4. Create a `.env` file in the root folder with the required API keys:
```python
ANTHROPIC_API_KEY=your_api_key_here
```

5. Launch ```main.py```

## Authors

Ole Kuhlmann  
Email: [ole.kuhlmann@rwth-aachen.de](mailto:ole.kuhlmann@rwth-aachen.de)  
GitHub: [ThatTilux](https://github.com/ThatTilux)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
