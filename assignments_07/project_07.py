# Part 2: Mini-Project - World Happiness Agent
"""
This project uses the World Happiness dataset — a set of yearly CSV files (2015-2024) 
reporting national happiness scores and the factors associated with them. You will use 
a CodeAgent to explore the data conversationally, rather than writing the analysis code yourself.
The goal is to build a complete agent from scratch: define the tools, instantiate the agent, and 
run it through a series of guided queries. Along the way you should see exactly where the agent uses 
your tools, where it writes its own code, and where its reasoning impresses or surprises you.
"""

# Task 1: Define Your Tools
from pathlib import Path
from scipy import stats
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# smolagents imports
from smolagents import CodeAgent, OpenAIServerModel, tool

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")
api_key = os.getenv("OPENAI_API_KEY")

RESOURCES_DIR = Path("happiness_project")

# Assign global dataframe
df = None 

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir

    # --- Small internal helpers -----------------
    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if df is None:
            return {
                "error": "No data is loaded yet.  Call load_happiness_data() first."
            }
        return None
    
    # --- Tools (public methods) -----------------
    # Tool 1: load_happiness_data
    def load_happiness_data(self) -> dict:
        """
        Load and merge every world_happiness_*.csv (2015-2024) in happiness_project/
        into the global df.
        """
        global df

        COLUMN_NAMES = {
            "Country": "country",
            "Regional indicator": "region",
            "Happiness score": "happiness_score",
            "Ladder score": "happiness_score",   # the 2024 file uses this name
            "GDP per capita": "gdp_per_capita",
        }

        paths = sorted(RESOURCES_DIR.glob("world_happiness_*.csv"))

        if not paths:
            return {
                "error": f"No world_happiness_*.csv files found in {self.resources_dir}/.",
                "available_files": self._available_csv_files(),
            }
        
        frames = []
        for path in paths:
            year_df = pd.read_csv(path, sep=";", decimal=",")
            year_df = year_df.rename(columns=COLUMN_NAMES)
            year_df["year"] = int(path.stem.split("_")[-1])   # "world_happiness_2020" -> 2020
            frames.append(year_df)

        df = pd.concat(frames, ignore_index=True)
    
        return {
            "message": f"Loaded data with shape {df.shape}.",
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "data": df,
        }

    # Tool 2: summarize_column
    def summarize_column(self, column: str) -> dict:
        """
        Return basic summary stats for a single column in the loaded dataset. 
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in df.columns:
            return {"error": f"Column '{column}' not found. Available columns are: {df.columns.tolist()}"}

        return df[column].describe(include="all").to_dict()

    # Tool 3: compute_correlation
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if col1 not in df.columns:
            return {"error": f"'{col1}' is not a column. Options: {df.columns.tolist()}"}
    
        if col2 not in df.columns:
            return {"error": f"'{col2}' is not a column. Options: {df.columns.tolist()}"}
    
        pearson = stats.pearsonr(df[col1], df[col2])
        result = {'col1': col1, 'col2': col2, 'pearson_r': float(pearson[0]), 'p_value': float(pearson[1])}
    
        cleaned = {}
        for key, value in result.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 4)
            else:
                cleaned[key] = value
    
        return cleaned

    # Tool 4: get_top_n_countries
    def get_top_n_countries(self, column: str, year: int, n: int = 5) -> dict:
        """Returns the top N countries ranked by a given column for a specific year.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in df.columns:
            return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}

        df_year = df[df["year"] == year]
        if df_year.empty:
            return {"error": f"No data for year '{year}'. Available years: {sorted(df['year'].unique().tolist())}"}

        top = df_year.nlargest(n, column)
        return [
            {"country": row["country"], column: getattr(row, column)}
            for _, row in top.iterrows()
        ]

csv_manager = CsvManager(resources_dir=RESOURCES_DIR)

# Tool 1: load_happiness_data
@tool
def load_happiness_data() -> dict:
    """Load and merge all yearly World Happiness files from happiness_project/.

    Reads every world_happiness_*.csv (2015-2024), normalizes the column names to
    country, region, happiness_score, gdp_per_capita, adds a year column, and stores
    the combined frame in the df

    Returns:
        A dict with keys "message", "shape", "columns", and "data", or an error dict.
        "data" is the merged dataframe, can be accessed with result["data"]
    """
    result = csv_manager.load_happiness_data()
    if "error" in result:
        return result
    result["data"] = df
    return result

# Tool 2: summarize_column
@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive statistics for a single column in the loaded world happiness dataset.

    Args:
        column: Name of the column to summarize, e.g., "happiness_score".

    Returns:
        A dict of summary statistics, or an error dict. 
    """
    return csv_manager.summarize_column(column)

# Tool 3: compute_correlation
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation between two columns in the loaded DataFrame.

    Args:
        col1: Name of first numeric column to correlate
        col2: Name of second numeric column to correlate

    Returns:
        Returns the correlation coefficient and p-value, or an error dict/string.
    """
    return csv_manager.compute_correlation(col1, col2)

# Tool 4: get_top_n_countries
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> list[dict]:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: Name of the column you want to rank
        year: Year of interest
        n: Number of countries to rank

    Returns:
        Returns a list of dicts of the top N countries. 
    """
    return csv_manager.get_top_n_countries(column, year, n)

# Task 2: Build the Agent
# Define tools for agent
TOOLS = [
    load_happiness_data,
    summarize_column,
    compute_correlation,
    get_top_n_countries,
]

# Define instructions for code agent
SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

# Initialize code agent
# Define model
model_to_use = "gpt-4o-mini"    # default model ID
model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas","matplotlib.pyplot","numpy","scipy.stats"],
    max_steps=8,
)

# Running the project
if __name__ == "__main__":
    # Task 3: Run Guided Queries
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = code_agent.run(query, reset=False)
        print(response)

    # Task 4: Your Own Questions
    # My query 1 - tool call 
    my_query_1 = "Show me the top 5 happiest countries in 2024. How did that change compared to the top 5 happiest countries in 2015?"
    response_1 = code_agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?
    """It was able to pull from the get_top_n_countries tool to compare the happiest countries in 2015 and 2024. """

    # My query 2 - both tool call and code generation
    my_query_2 = "Show me the bottom 5 happiest countries in 2024. How did that change compared to the bottom 5 happiest countries in 2015?"
    response_2 = code_agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    """It triggered both tool use and code generation. It started with the get_top_n_countries, but did generate 
    some code to get the bottom countries."""

# Task 5: Reflection
# 1. In Query 3, how did the agent report the correlation? The `compute_correlation` tool
#    returns both a Pearson r and a p-value — did the agent compute and communicate the
#    result sensibly (for example, describing the strength and direction of the
#    relationship, and noting how confident we can be in it)?
"""
The final answer was: {'correlation_coefficient': 0.6313, 'p_value': 0.0, 'statistically_significant': True}.
It did not give additional context, just computed the results. It would be helpful to update the prompt so that
the agent knows to communicate the results sensibly, describing things like the strength, direction of the relationship,
and confidence in the computations. 
"""
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
"""
I was surprised by the responses to my last two questions, "Show me the top 5 happiest countries in 2024. How did that 
change compared to the top 5 happiest countries in 2015?" and "Show me the bottom 5 happiest countries in 2024. How did 
that change compared to the bottom 5 happiest countries in 2015?". Not only did the agent pull both sets, but also compared
the changes from 2015 to 2024 for both the bottom and top 5 happiest countries. That was more helpful than just pulling the lists 
of each, saving the user the time of having to compare the two. 
"""
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
"""
It would be helpful to add one additional tool that creates at least one type of visualization (scatter plot, bar chart, box plot).
Visualizations allow you to quickly visualize the data, allowing for further data insights. Visualizations could help the 
user answer a multitude of questions. With scatter plots, users are able to more easily analyze the relationship between two variables. 
With bar charts, users are able to easily compare numerical values in the data. With box plots, users are more easily able to see
the min, max, and summary statistics of a column in a dataset. So, many questions could be answered like "How does the GDP impact
the happiness score of a country?" (scatter), "How many countries have a happiness score of 1, 2, 3, ....?" (bar chart), "What is the 
minimum happiness score? What is the maximum happiness score?" (box plot). These are just a few questions that could be explored through
visualizations, but a visualization tool would open the door to allow the user to explore the data visually. 
"""